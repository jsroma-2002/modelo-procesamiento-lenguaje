# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
EXPOSE 80

RUN pip install -r requirements.txt
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader -d /usr/local/share/nltk_data punkt

EXPOSE 5000

# Run chat.py when the container launches
CMD ["python", "chat.py"]