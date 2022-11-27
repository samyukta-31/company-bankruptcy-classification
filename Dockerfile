# syntax=docker/dockerfile:1
FROM ubuntu:18.04
LABEL maintainers="adharsh.venkat98@gmail.com, sammie1999@gmail.com"
LABEL version="1.0"
LABEL description="This is custom Docker Image for \
the Samyukta and Adharsh's Company Bankruptcy Classification ML Project."
SHELL ["bash", "-c"]
WORKDIR /company-bankruptcy-classification
COPY . .
RUN apt update && \
    apt install python3-pip -y
RUN pip install -r requirements.txt
RUN python3 main.py

CMD ["bash"]
EXPOSE 3000