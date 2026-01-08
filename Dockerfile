FROM python:3.13
WORKDIR /usr/local/app

# Install the application dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy in the source code
COPY predict.py ./predict.py
COPY loan_default_model.pkl ./loan_default_model.pkl
EXPOSE 5000

# Setup an app user so the container doesn't run as the root user
RUN useradd myuser
USER myuser

CMD ["python", "predict.py"]

# docker build -t rabishankarsahu/mymodel .
# docker push rabishankarsahu/mymodel:latest
# docker pull rabishankarsahu/mymodel:latest
# docker run -p 5000:5000 rabishankarsahu/mymodel:latest
