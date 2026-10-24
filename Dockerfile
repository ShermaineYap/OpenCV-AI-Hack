# One image for local use and AWS Lambda (via the AWS Lambda Web Adapter).
FROM public.ecr.aws/docker/library/python:3.13-slim

# Lambda Web Adapter: lets the unmodified FastAPI app run on Lambda behind a Function URL.
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.9.1 /lambda-adapter /opt/extensions/lambda-adapter

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    AWS_LWA_READINESS_CHECK_PATH=/health \
    BOILGUARD_RUNS_DIR=/tmp/boilguard-runs \
    OPENCV_LOG_LEVEL=ERROR

WORKDIR /var/task
COPY requirements.txt .
RUN pip install -r requirements.txt && python -c "import cv2; assert cv2.__version__.startswith('5'), cv2.__version__; print('OpenCV', cv2.__version__)"

COPY boilguard ./boilguard
COPY app ./app
COPY results ./results

RUN useradd -m appuser
USER appuser
EXPOSE 8080
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
