defmodule AgenticRuntime do
  @moduledoc """
  Documentation for `AgenticRuntime`.
  """

  alias AgenticRuntime.Agents.Factory

  defdelegate build_anthropic_model_config(model_name, api_key, opts), to: Factory
  defdelegate build_openai_model_config(model_name, api_key), to: Factory
  defdelegate build_googleai_model_config(model_name, api_key), to: Factory

  @doc """
  Required opts: 
  * model_config
  * base_system_prompt
  * tools 
  """
  defdelegate create_agent(opts), to: Factory

  defdelegate build_new_user_message!(message_text), to: LangChain.Message, as: :new_user!
  defdelegate add_message(agent_id, langchain_message), to: Sagents.AgentServer
  defdelegate cancel_agent_execution(agent_id), to: Sagents.AgentServer, as: :cancel

  # Channel helpers
  # See AgenticRuntime.IntegrationHelpers
end
