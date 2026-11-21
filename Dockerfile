FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    OPENBLAS_NUM_THREADS=1
WORKDIR /opt/sparse

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY cli.py LICENSE NOTICE ./
COPY orbitsight/*.py orbitsight/
COPY scripts/make_viz.py scripts/eval_ebssa.py scripts/

USER 65534:65534
ENTRYPOINT ["python", "cli.py"]
CMD ["--help"]
