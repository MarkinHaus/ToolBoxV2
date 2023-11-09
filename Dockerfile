# Use an official Python runtime as a parent image
FROM python:3.9-slim

MAINTAINER Markin Hausmanns MarkinHausmanns@gmail.com

# Update aptitude with new repo
RUN apt-get update

# Install software
RUN apt-get install -y git

# Set the working directory in the container to /app
WORKDIR /app

# Clone the git repo into the docker container
RUN git clone -b light-tb https://github.com/MarkinHaus/ToolBoxV2.git

#WORKDIR /ToolBoxV2/


# Install any needed packages specified in requirements.txt
RUN pip install -e ./ToolBoxV2/

#RUN npm install ./ToolBoxV2/toolboxv2/app/node_modules/.package-lock.json
# Make port 5000 available to the world outside this container
EXPOSE 5000:5000

# Run the command to start the toolbox -n modInstaller
CMD ["toolboxv2", "-m", "api", "-n", "live"]
