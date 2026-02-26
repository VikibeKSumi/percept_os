FROM ultralytics/ultralytics:latest

WORKDIR /app

COPY . /app

RUN pip install -e .

CMD ["python", "-m", "percept_os.cli.run", "jobs/edge_demo.json"]