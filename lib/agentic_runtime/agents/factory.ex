defmodule AgenticRuntime.Agents.Factory do
  @moduledoc """
  Factory for creating agents with consistent configuration.

  This module centralizes agent creation, ensuring all agents use the same
  model, middleware stack, and base configuration. The Coordinator calls
  `create_agent/1` when starting a conversation session.

  This Factory is automatically configured for your persistence layer:
  - Owner type: :user
  - Owner field: user_id
  - Conversations context: AgenticRuntime.Conversations

  ## Customization

  - Change model provider in `get_model_config/0`
  - Configure fallbacks in `get_fallback_models/0`
  - Configure title generation model in `get_title_model/0`
  - Modify system prompt in `base_system_prompt/0`
  - Add/remove middleware in `build_middleware/2`
  - Add custom tools in `create_agent/1` under the `:tools` key
  - Configure HITL in `default_interrupt_on/0`

  ## Understanding the Default Middleware

  The middleware stack below replicates `Sagents.Agent.build_default_middleware/3`.
  You can call that function in IEx to see the canonical defaults:

      middleware = Sagents.Agent.build_default_middleware(model, "test-agent")

  ## Model Fallback Strategy

  The fallback configuration uses the *same model* on a different provider
  for resilience without changing behavior:

  | Primary Provider      | Fallback Provider       |
  |-----------------------|-------------------------|
  | ChatAnthropic (API)   | ChatAnthropic (Bedrock) |
  | ChatOpenAI (API)      | ChatOpenAI (Azure)      |

  ## Filesystem Scoping

  The FileSystem middleware supports flexible scoping to control file isolation.

  **For user-facing interactive agents (recommended):**
  - **User-scoped**: `filesystem_scope: {:user, user_id}`
    - Files persist across all conversations for the same user
    - Enables true long-term memory and file accumulation
    - This is the typical pattern for chat applications

  **Other scoping options:**
  - **Project-scoped**: `filesystem_scope: {:project, project_id}`
    - Files shared within a project across multiple users and conversations

  - **Agent-scoped**: `filesystem_scope: nil` (defaults to `{:agent, agent_id}`)
    - Files isolated per conversation - typically too limiting for user-facing agents
    - May be appropriate for non-interactive batch processing or isolated task execution

  - **Custom scoping**: Use any tuple like `{:team, team_id}` or `{:session, session_id}`

  **Important:** The Coordinator should pass the appropriate scope when calling `create_agent/1`.

  """

  alias LangChain.ChatModels.ChatAnthropic
  alias LangChain.ChatModels.ChatGoogleAI
  alias LangChain.ChatModels.ChatOpenAI
  alias Sagents.Agent
  alias Sagents.Middleware.ConversationTitle
  alias Sagents.Middleware.HumanInTheLoop

  @doc """
  Creates an agent with the standard configuration.

  ## Options

  - `:agent_id` - Required. Unique identifier for this agent.
  - `:filesystem_scope` - Optional. Scope tuple for filesystem isolation.
    Examples: `{:user, user_id}`, `{:project, 456}`, `{:team, 789}`.
    Pass `nil` for agent-scoped (isolated per conversation). This is the 
    default behaviour.
  - `:interrupt_on` - Optional. Map of tool names requiring approval.
    Pass `nil` to disable HITL entirely.

  ## Examples

      # Standard usage with user scope
      {:ok, agent} = Factory.create_agent(
        agent_id: "conv-123",
        filesystem_scope: {:user, user_id}
      )

      # Project-scoped filesystem
      {:ok, agent} = Factory.create_agent(
        agent_id: "conv-123",
        filesystem_scope: {:project, project_id}
      )

      # Agent-scoped (isolated per conversation)
      {:ok, agent} = Factory.create_agent(
        agent_id: "conv-123",
        filesystem_scope: nil
      )

      # With custom HITL configuration
      {:ok, agent} = Factory.create_agent(
        agent_id: "conv-123",
        filesystem_scope: {:user, user_id},
        interrupt_on: %{
          "write_file" => true,
          "delete_file" => true,
          "execute_command" => true
        }
      )

  """
  def create_agent(opts \\ []) do
    agent_id = Keyword.fetch!(opts, :agent_id)
    filesystem_scope = Keyword.get(opts, :filesystem_scope, nil)
    interrupt_on = Keyword.get(opts, :interrupt_on, default_interrupt_on())
    main_model_config = Keyword.fetch!(opts, :model_config)
    title_model_config = Keyword.get(opts, :title_model_config, main_model_config)
    base_system_prompt = Keyword.fetch!(opts, :base_system_prompt)
    tools = Keyword.get(opts, :tools, [])
    fallback_models = Keyword.get(opts, :fallback_models, [])
    before_fallback = Keyword.get(opts, :before_fallback, nil)
    tool_context = Keyword.get(opts, :tool_context, %{})

    Agent.new(
      %{
        agent_id: agent_id,
        model: main_model_config,
        base_system_prompt: base_system_prompt,
        middleware: build_middleware(filesystem_scope, interrupt_on, title_model_config),
        fallback_models: fallback_models,
        before_fallback: before_fallback,
        tools: tools,
        tool_context: tool_context
      },
      # Since we specify the full middleware stack, don't add defaults
      replace_default_middleware: true
    )
  end

  # ---------------------------------------------------------------------------
  # Model Configuration
  # ---------------------------------------------------------------------------
  def build_anthropic_model_config(model_name, api_key, opts \\ []) do
    thinking_opts = Keyword.get(opts, :thinking, %{type: "enabled"})

    ChatAnthropic.new!(%{
      model: model_name,
      api_key: api_key,
      stream: true,
      thinking: thinking_opts
    })
  end

  def build_openai_model_config(model_name, api_key) do
    ChatOpenAI.new!(%{
      model: model_name,
      api_key: api_key,
      stream: true
    })
  end

  def build_googleai_model_config(model_name, api_key) do
    ChatGoogleAI.new!(%{
      model: model_name,
      api_key: api_key,
      stream: true
    })
  end

  # ---------------------------------------------------------------------------
  # Human-in-the-Loop Configuration
  # ---------------------------------------------------------------------------

  # Default tools that require human approval before execution.
  # Return `nil` or `%{}` to disable HITL entirely.
  #
  # Configuration options:
  #   - `true` - Enable with default decisions (approve, edit, reject)
  #   - `false` - No interruption for this tool
  #   - `%{allowed_decisions: [:approve, :reject]}` - Custom decisions
  #
  defp default_interrupt_on do
    nil
  end

  # ---------------------------------------------------------------------------
  # Middleware Configuration
  # ---------------------------------------------------------------------------

  # Build the middleware stack.
  #
  # This replicates the default stack from `Sagents.Agent.build_default_middleware/3`:
  #   1. TodoList - Task management with write_todos tool
  #   2. ConversationTitle - Auto-generate conversation titles (async, so positioned early)
  #   3. FileSystem - Virtual filesystem (ls, read_file, write_file, etc.)
  #   4. SubAgent - Delegate to specialized child agents
  #   5. Summarization - Compress long conversations to stay within token limits
  #   6. PatchToolCalls - Fix dangling tool calls from interrupted conversations
  #
  # HumanInTheLoop is conditionally added based on `interrupt_on` configuration.
  #
  # Order matters! Early middleware sees messages first (before_model) and
  # processes responses last (after_model).
  #
  defp build_middleware(filesystem_scope, interrupt_on, title_model_config) do
    [
      # Task management - gives the agent a todo list for tracking work
      Sagents.Middleware.TodoList,

      # ConversationTitle - auto-generate titles after the first exchange.
      # Positioned early because it's async and should start as soon as possible.
      # Uses a lighter/faster model (Haiku) for cost efficiency.
      {ConversationTitle,
       [
         chat_model: title_model_config,
         fallbacks: []
       ]},

      # Virtual filesystem - file operations with configurable scope
      # For user-facing agents, pass {:user, user_id} from the Coordinator
      # If nil, defaults to {:agent, agent_id} (isolated per conversation)
      {Sagents.Middleware.FileSystem, [filesystem_scope: filesystem_scope]},

      # SubAgent - spawn child agents for complex tasks
      # Configure block_middleware to prevent certain middleware from being
      # inherited by subagents (e.g., Summarization, ConversationTitle).
      {Sagents.Middleware.SubAgent,
       [
         block_middleware: [
           Sagents.Middleware.Summarization,
           Sagents.Middleware.ConversationTitle
         ]
       ]},

      # Summarization - compress long conversations to fit context window
      Sagents.Middleware.Summarization,

      # PatchToolCalls - fix dangling tool calls from interrupted conversations
      Sagents.Middleware.PatchToolCalls
    ]
    # Conditionally add HumanInTheLoop if interrupt_on is configured.
    |> HumanInTheLoop.maybe_append(interrupt_on)
  end
end
