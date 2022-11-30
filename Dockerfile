# syntax=docker/dockerfile:1
FROM ubuntu:18.04
LABEL maintainers="adharsh.venkat98@gmail.com, sammie1999@gmail.com"
LABEL version="1.0"
LABEL description="This is custom Docker Image for \
the Samyukta and Adharsh's Company Bankruptcy Classification ML Project."

USER root

SHELL ["bash", "-c"]

WORKDIR /company-bankruptcy-classification
COPY . .

RUN apt update && \
    apt install python3-pip -y && \
    pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

RUN touch /ml_pipeline.sh
RUN echo "#!/bin/sh" >> /ml_pipeline.sh
RUN echo "cd company-bankruptcy-classification" >> /ml_pipeline.sh
RUN echo "python3 preprocessing.py" >> /ml_pipeline.sh
RUN chmod +x /ml_pipeline.sh

CMD ["bash"]
