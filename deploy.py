from azure.ai.ml import MLClient, load_model, load_environment, load_online_endpoint, load_online_deployment
from azure.ai.ml.entities import Model, Environment, ManagedOnlineDeployment, ManagedOnlineEndpoint
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import HttpResponseError, ServiceRequestError
from typing import Annotated, Tuple
from enum import Enum
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.logging import RichHandler
from rich.console import Console
import typer
import logging
import requests

class LoggingLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

app = typer.Typer(no_args_is_help=True)
logger = logging.getLogger(__name__)
console = Console()

# def setup_logging(verbosity: str) -> None:
#     """Set up logging based on verbosity level."""
#     log_level = getattr(logging, verbosity.upper(), logging.WARNING)
#     logging.basicConfig(
#         level=log_level,
#         format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#         handlers=[RichHandler()]
#     )
#     logger.info(f"Logging level set to {verbosity.upper()}")

def load_configuration_files() -> Tuple[Model, Environment, ManagedOnlineEndpoint, ManagedOnlineDeployment]:
    """Load AzureML objects from .yaml-files"""
    model = load_model("azureml/iris-model.yaml")
    environment = load_environment("azureml/iris-model-environment.yaml")
    endpoint = load_online_endpoint("azureml/iris-model-endpoint.yaml")
    deployment = load_online_deployment("azureml/iris-model-deployment.yaml")
    return model, environment, endpoint, deployment

def test_endpoint(
        ml_client: MLClient, 
        deployment: ManagedOnlineDeployment,
        endpoint: ManagedOnlineEndpoint
    ) -> None:
    """Sends a request to the endpoint in order to confirm deployment."""
    with console.status("Testing endpoint..", spinner="arc"):
        console.log(f"Testing endpoint: {deployment.endpoint_name}..")

        try:
            endpoint_name = endpoint.name
            scoring_uri = endpoint.scoring_uri
            key = ml_client.online_endpoints.get_keys(endpoint_name).primary_key

            sample_input = {
                "data": [
                    [4.6, 3.6, 1.0, 0.2],
                    [7.2, 3.1, 5.8, 2.1]
                ]
            }

            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {key}',
                'azureml-model-deployment': deployment.name
            }

            response = requests.post(scoring_uri, json=sample_input, headers=headers)
            
            if response.status_code == 200:
                console.print_json(f"Endpoint test successful! Response: {response.json()}")
            else:
                console.print_exception(f"Endpoint test failed with status code: {response.status_code}")
                console.log(f"Response: {response.text}")
                raise typer.Exit(code=1)

        except Exception as e:
            console.log(f"Error testing endpoint: {str(e)}")
            raise typer.Exit(code=1)

@app.command()
def main(
    subscription_id: Annotated[str, typer.Option(help="The Azure subscription ID.")],
    resource_group_name: Annotated[str, typer.Option(help="The name of the resource group.")],
    workspace_name: Annotated[str, typer.Option(help="The name of the AzureML workspace.")],
) -> None:
    """
    This command-line script demonstrates how to deploy the example model to an Azure cloud environment,
    specifically using AzureML and Managed Endpoints.
    """

    #setup_logging(verbose.value)
    console.log("Initializing MLClient...")

    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group_name,
        workspace_name=workspace_name,
    )
    try:
        with console.status("Loading configuration files..", spinner="arc") as status:
            model, environment, endpoint, deployment = load_configuration_files()
            console.log("Loaded .yaml-files..")
            
            status.update(status="Creating model..", spinner="arc")
            model_result = ml_client.models.create_or_update(model)
            console.log(f"Model created: {model_result.name}")

            status.update(status="Creating environment..", spinner="arc")
            environment_result = ml_client.environments.create_or_update(environment)
            console.log(f"Environment created: {environment_result.name}")

            status.update(status="Creating endpoint..", spinner="arc")
            endpoint_result = ml_client.online_endpoints.begin_create_or_update(endpoint).result()
            console.log(f"Endpoint created: {endpoint_result.name}")

            status.update(status="Creating deployment..", spinner="arc")
            deployment_result = ml_client.online_deployments.begin_create_or_update(deployment).result()
            console.log(f"Deployment created: {deployment_result.name}")
            console.log("Deployment completed successfully!")

    except HttpResponseError as e:
        console.print_exception(f"Azure service error: {e.message}")
        raise typer.Exit(code=1)

    except ServiceRequestError as e:
        console.print_exception(f"Request error: {e.message}")
        raise typer.Exit(code=1)

    except Exception as e:
        console.print_exception(f"Unexpected error: {str(e)}")
        raise typer.Exit(code=1)
    console.rule("Endpoint testing")
    test_endpoint(ml_client, deployment_result, endpoint_result)

if __name__ == "__main__":
    app()