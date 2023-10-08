# https://www.uvicorn.org/settings/
# Same creds as: https://github.com/dkmiller/odin/blob/main/.github/workflows/build.yml
AZURE_TENANT_ID=$(op read op://Private/github-actions/tenant) \
  AZURE_CLIENT_ID=$(op read op://Private/github-actions/username) \
  AZURE_CLIENT_SECRET=$(op read op://Private/github-actions/credential) \
  uvicorn main:app --reload --reload-exclude test_*
