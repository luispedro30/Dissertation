FROM python:3.10.7
WORKDIR /app
COPY . /app
RUN apt-get update && apt-get install -y portaudio19-dev
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8080
CMD python ./app.py