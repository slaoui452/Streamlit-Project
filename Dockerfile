From python:3.8
WORKDIR / streamlit-app
COPY requirements.txt .
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"
RUN apt update --allow-insecure-repositories
RUN apt-get install -y aptitude
RUN aptitude install -y libjasper1 libjasper-dev
RUN pip install -r requirements.txt
EXPOSE 8083
COPY ./app ./app
ENTRYPOINT ["streamlit", "run", "app/main.py", "--server.port=8083", "--server.address=0.0.0.0"]