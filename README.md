### 1. Write App (Flask, TensorFlow)
- The code to build, train, and save the model is in the `test` folder.
- Implement the app in `main.py`
### 2. Setup Google Cloud 
- Create new project
- Activate Cloud Run API and Cloud Build API

### 3. Install and init Google Cloud SDK
- https://cloud.google.com/sdk/docs/install

### 4. Dockerfile, requirements.txt, .dockerignore
- https://cloud.google.com/run/docs/quickstarts/build-and-deploy#containerizing

### 5. deploy
```
pip freeze > requirements.txt
gcloud builds submit --tag gcr.io/emgmlrecognition/<image_name>
gcloud run deploy --image gcr.io/emgmlrecognition/<image_name> --platform managed


for explain url gcr.io/project_id/function_name 
function name is cloud run name 
first data build docker image  send for 
gcloud builds submit --tag gcr.io/emgmlrecognition/emgfull
gcloud run deploy --image gcr.io/emgmlrecognition/emgfull --platform managed
```
# 5. Cloud build & auto deploy 
### Cell 1: Title

```
# Cloud Build Configuration Breakdown

```
### Cell 2: Introduction

```
This notebook explains the provided Cloud Build configuration for building, pushing, and deploying a Docker image to Cloud Run.

```
### Cell 3: Steps

The configuration consists of several steps:

#### 3.1 (Optional) Maven Build

```
#This step is currently commented out

- name: 'maven'
  entrypoint: 'mvn'
  args: ['clean', 'install', '-DskipTests']
This step uses the maven builder to execute Maven commands (if your project uses Maven).
It's commented out by default. Uncomment if you need to clean the project, install dependencies, and skip tests before building the image.

```
#### 3.2 Build Docker Image

```
- name: 'gcr.io/cloud-builders/docker'
  args: ['build', '-t', 'gcr.io/$PROJECT_ID/$_SERVICE_NAME:$COMMIT_SHA', '.']
This step uses the gcr.io/cloud-builders/docker builder to build the Docker image.
Arguments:
build: Instructs the builder to build.
-t: Specifies the target image tag:
gcr.io/$PROJECT_ID/$_SERVICE_NAME:$COMMIT_SHA
$PROJECT_ID: Replaced with your GCP project ID at build time.
$_SERVICE_NAME: Replaced with the name of your Cloud Run service at build time.
$COMMIT_SHA: Replaced with the Git commit SHA of the build at build time. This ensures unique image tags for each build.
.: Indicates the current directory as the build context (where your Dockerfile resides).

```
#### 3.3 Push Docker Image

```
- name: 'gcr.io/cloud-builders/docker'
  args: ['push', 'gcr.io/$PROJECT_ID/$_SERVICE_NAME:$COMMIT_SHA']
This step uses the same docker builder to push the built image to Container Registry (GCR).
Arguments:
push: Instructs the builder to push.
gcr.io/$PROJECT_ID/$_SERVICE_NAME:$COMMIT_SHA: The same image tag used during build.

```
#### 3.4 Deploy to Cloud Run

```
- name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
  entrypoint: gcloud
  args:
    - 'run'
    - 'deploy'
    - '$_SERVICE_NAME'
    - '--image'
    - 'gcr.io/$PROJECT_ID/$_SERVICE_NAME:$COMMIT_SHA'
    - '--region'
    - '$_DEPLOY_REGION'
This step uses the cloud-sdk builder to deploy the image to Cloud Run.
Arguments:
gcloud: Invokes the Cloud SDK.
run: Interacts with Cloud Run services.
deploy: Deploys a service.
$_SERVICE_NAME: Replaced with the service name at build time.
--image: Specifies the image to deploy (same tag used before).
--region: Sets the deployment region (replace $_DEPLOY_REGION with your desired region at build time).

```
### Cell 4: Logging and Images

```
options: Configures build options.
logging: CLOUD_LOGGING_ONLY: Instructs Cloud Build to send logs only to Cloud Logging (not Stackdriver stdout/stderr).
images: Declares the image built during the process (for reference).

```
### Cell 5: Conclusion

```
This Cloud Build configuration automates the CI/CD pipeline for your Cloud Run service. It dynamically creates image tags and deploys to the specified region based on environment variables. Remember to set appropriate values for $PROJECT_ID and $_DEPLOY_REGION.

```
#start practical process step 

