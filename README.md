# Description
In this repository I demonstrate how to deploy a machine learning model to an AzureML Managed Endpoint. I developed a deployment script that uses the [AzureML Python SDK]() as well as [Typer]() to create a modern command-line interface for the deployment script. 

## Initial challenges
I initially faced some challenges due to the lack of an environment specification, which is typically packaged together with a machine learning model. Furthermore, the scoring script was not correctly setup. I managed to resolve the issues, but they did reduce the scope of what I intended to achieve.

## AzureML Managed Endpoints
The [AzureML]()-suite provides certain capabilities such as the [Model Registry]() and [Managed Endpoints](), which simplify the management and deployment of machine learning models to an Azure cloud environment.

Specifically this solution offers:
- CPU-, time- or metric-based auto-scaling rules
- Separation of concerns between managing infrastructure and deploying ML models
- Safe rollout options such as blue/green deployment
- Continuos monitoring and logging of endpoints for detailed debugging and error handling


## Architecture Diagram
The following architecture diagram details the solution that I aimed for:

![](/assets/architecture-diagram.png)


## Deployment flow
In order to deploy the model, I take the following steps:
1. Register the model in the AzureML Model Registry
2. Register the environment in the AzureML Workspace
3. Create the endpoint
4. Create the specific deployment

If you want to reproduce what the deployment script does manually, you can run the following commands:
```sh
# Create the model and register it
az ml model create -f iris-model.yaml

# Create environment 
az ml environment create -f iris-model-environment.yaml

# Create endpoint for deployment
az ml online-endpoint create -f iris-model-endpoint.yaml

# Create deployment
az ml online-deployment create --endpoint sklearn-test-endpoint -f iris-model-deployment.yaml
```

In order to setup Python run:
`python -m venv .venv`
`python -m pip install --upgrade pip`
`pip install -r requirements.txt`

## Future work
While this deployment showcases how to deploy a machine learning model to the cloud, it does not demonstrate best practices in terms of neither MLOps nor DevOps. Therefore I will use this section to reflect about which changes could be made to create a robust and scalable architecture.

### 1. Environment Promotion
All entities such as models, environments, deployments and endpoints should be promoted through their respective environments for quality assurance. This should be done through the CI/CD pipeline.

### 2. Monitoring and logging
Using the built-in integration between AzureML Managed Endpoints and Azure Application Insights / Log Analytics / Monitor it is possible to monitor the endpoints and models in real-time. This would allow for a more robust diagnostics capability as well as the possibility to trigger alerts or warnings when certain conditions were met such as model drift.

### 3. AzureML Model registry
By using the AzureML workspace as a registry for models, it is possible to use the registry as a separation-of-concern mechanism. This allows data scientists to train and develop their models independently and submit them to the registry when they are ready to deploy them. By using Azure Event Grid it is possible to trigger a deployment script that would then promote newly submitted model through the environments. 

### 4. Azure API Management
Using API management to manage the APIs between the models and their endpoints improves the robustness of the solution, because it separates the responsibility of the endpoint technicalities from the AzureML Managed Endpoint into Azure API Management. Some of the added benefits include:
- Create and manage endpoints across multiple models
- Use subscriptions and subscription keys to control access to different users and groups
- Use backend policies to determine how API calls should be managed and routed

# Scaling the microservices architecture
At the end of the day I propose that a microservices architecture is the central tenet of enterprise-scale machine learning models. Utilizing libraries such as [LitServe]() it can provide the basis for serving custom docker images at scale, that perform according to the requirements of the specific implementation. It is meaningful to stick with AzureML Managed Endpoints and all the managed services they offer, but it is also viable to move to a more serverless architecture using Azure Container Apps.

Ultimately it is about balancing complexity, technical debt with scalability, cost-effectivity and performance.
