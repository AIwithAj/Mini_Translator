# Minit-Translator

In my project building a mini Google translator using sequence-to-sequence models
										with PyTorch from scratch, I first loaded the dataset from Hugging Face. Then, I
										constructed the sequence-to-sequence architecture and conducted training. During
										training, I achieved an accuracy of 85% on the training dataset and 82% on the
										validation dataset.
										Subsequently, I developed my own Transformer architecture entirely from scratch
										using PyTorch. This involved implementing components such as the encoder,
										decoder, self-attention, and multi-head attention. Upon applying this custom
										Transformer architecture to the Hugging Face dataset, I obtained an accuracy of
										92% on the training set and 90% on the validation set during training.
										Please note that these accuracy values are placeholders and can be replaced with
										actual performance metrics obtained during your experimentation.
									

# End-to-End-NLP-Project-using-weights and biases -DVC


## Workflows

1. Update config.yaml								    # storing the constants
2. Update secrets.yaml [Optional]
3. Update params.yaml                        	    	# storing the parameters constants
4. Update the entity     								# output for the next component
5. Update the configuration manager in src config      # here we are creating paths (merging config and entity)
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the dvc.yaml







### DVC cmd

1. dvc init
2. dvc repro
3. dvc dag


## About   DVC



DVC 

 - Its very lite weight for POC only
 - lite weight expriements tracker
 - It can perform Orchestration (Creating Pipelines)



# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
    - Save the URI: 566373416292.dkr.ecr.us-east-1.amazonaws.com/chicken

	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


# 7. Setup github secrets:

    AWS_ACCESS_KEY_ID=

    AWS_SECRET_ACCESS_KEY=

    AWS_REGION = us-east-1

    AWS_ECR_LOGIN_URI = demo>>  566373416292.dkr.ecr.ap-south-1.amazonaws.com

    ECR_REPOSITORY_NAME = simple-app

