defmodule AgenticRuntime do
  @moduledoc """
  Documentation for `AgenticRuntime`.
  """

  alias AgenticRuntime.Agents.Factory

  defdelegate build_anthropic_model_config(model_name, api_key, opts), to: Factory
  defdelegate build_openai_model_config(model_name, api_key), to: Factory
  defdelegate build_googleai_model_config(model_name, api_key), to: Factory

  defdelegate create_agent(opts), to: Factory

  # Channel helpers
  # See AgenticRuntime.IntegrationHelpers
end
