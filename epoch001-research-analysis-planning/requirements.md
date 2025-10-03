# Project Requirements

We are going to create a proad scope predictive maintainance demonstration example.  That way we can use the same models and other assets but then just create new front-end applications based on the use case and industry.

---

## Epoch 001 - Planning

Create a comprehensive plan with desired deliverables.  The other agents will use this to plan and execute their individual work.

## Epoch 002 - Data

Include real world data types from different sensors time series and image analysis.  See if there is any visual information from sites like Kagle (https://www.kaggle.com/) that you can download.

Include data types that would be realistic for the military (Army, Navy, Air Force, Marines), Oil and Gas (offshore, drilling, pumping, refineing, etc) Logistics (Trucking, Warehouse automation, robotices) and manufacturing (industrial, cpg, production line). and other appropriate 


## Epoch 003 - Exploratory Analysis

Generate multiple notebooks based on the industries mentioned above.

## Epoch 004 - Feature Engineering

Do this as instructed

## Epoch 005 - Model Development

Train and Tune the models as instructed

## Epoch 006 - Model Testing

Use your best judgement

## Epoch 007 - Deployment & App

- Lets start with an example of a single Streamlet application for the manufacturing example.  We can add more later.
- For that app, we should have 3 tabs:
    - Executive Overview: includes high level broad time frame view of the overall situation with heat charts that show hot spots
    - Operations: Realt Time detailed fiew of items that are critical and potentially going to fail soon.
    - Insight:  A report that provides insights and recommendations.  Leverage the Domino Data Lab LLM Gateway to connect to ChatGPT to develop a PDF-based recommendations report.

```
from mlflow.deployments import get_deploy_client
import os

client = get_deploy_client(os.environ['DOMINO_MLFLOW_DEPLOYMENTS'])

response = client.predict(
	endpoint="ChatGPT-4",
	inputs={"messages": [{"role": "user", "content": "Tell me a joke about rabbits"}]}
)
print(response)
```

## Epoch 008 - Retrospective

For the retrospective, review places where we had errors or had to troubleshoot items and make suggestions on what we can modify with the agents to make them perform more efficiently in the future.

---

**Instructions for Business Analyst**: Read this file first. Ask the user for clarification on anything unclear before proceeding.
