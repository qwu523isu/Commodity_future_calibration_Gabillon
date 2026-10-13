FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements-risk.txt ./
RUN python -m pip install --no-cache-dir -r requirements-risk.txt

COPY risk_api.py Step4a_minimal_risk_engine.py Step4b_portfolio_risk_engine.py ./
COPY Input_data/risk_engine_config.json ./Input_data/
COPY Output/Step1a_EIA_Preparation/EIA_WTI_calibration_input.csv ./Output/Step1a_EIA_Preparation/
COPY Output/Step1c_EIA_Full_Sample/EIA_calibration_results.csv ./Output/Step1c_EIA_Full_Sample/
RUN mkdir -p /data && chown 10001:10001 /data
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).close()"
ENTRYPOINT ["python", "risk_api.py", "--host", "0.0.0.0", "--output", "/data"]
CMD ["--port", "8000"]
