# Deploy to Azure Container Apps

What you end up with: a public HTTPS URL for the API that **scales to zero** (you pay nothing for compute while
nobody uses it), an image registry, and a GitHub Actions pipeline that redeploys on every push to `main`.

```
GitHub (push to main) --OIDC--> Azure
   test -> build -> smoke test -> push image to ACR -> new Container App revision -> smoke test the URL
                                                  secrets live ONLY in the Container App
```

Everything below is run by **you**, once, in your terminal. Nothing in this repository creates Azure resources by
itself. Keep the same terminal window open from step 1 to step 6 (the variables live in it). If you close it,
re-run step 1.

Rough cost: Container Apps (Consumption plan) has a monthly free allowance and costs nothing while scaled to
zero; Azure Container Registry **Basic is about US$5 per month**; Log Analytics is pay-per-GB with a free
monthly allowance; each `/ask` costs about US$0.002 in Azure OpenAI tokens. Check current prices on the Azure
pricing pages. Step 10 deletes everything.

## 0. Tools, login, providers

```bash
brew install azure-cli                       # skip if `az version` already works
az login                                     # opens a browser
az account show --query "{name:name, id:id, tenant:tenantId}" -o table   # is this the right subscription?
# if not:  az account set --subscription "<name or id>"

az extension add --name containerapp --upgrade --yes
az provider register --namespace Microsoft.App --wait
az provider register --namespace Microsoft.OperationalInsights --wait
az provider register --namespace Microsoft.ContainerRegistry --wait
```

## 1. Names (the only thing you choose)

```bash
export RG=fmcg-copilot-rg
export LOCATION=swedencentral
export ACR_NAME=fmcgcopilot$(openssl rand -hex 3)     # globally unique, lowercase letters and digits only
export ENV_NAME=fmcg-copilot-env
export APP_NAME=fmcg-copilot-api
export IDENTITY_NAME=fmcg-copilot-acr-pull
export GH_REPO=abdelrahmansharafx/fmcg-sales-copilot  # owner/repo on GitHub

echo "ACR_NAME=$ACR_NAME"                             # write this down: GitHub needs it in step 7
az group show -n $RG --query "{name:name, location:location}" -o table   # your existing resource group
```

## 2. Container registry (Basic)

```bash
az acr create -g $RG -n $ACR_NAME --sku Basic -l $LOCATION --admin-enabled false
```

`--admin-enabled false`: nobody gets a registry password; access is by Azure identity only.

## 3. An identity that may pull images (no password)

```bash
az identity create -g $RG -n $IDENTITY_NAME -l $LOCATION
export IDENTITY_ID=$(az identity show -g $RG -n $IDENTITY_NAME --query id -o tsv)
export IDENTITY_PRINCIPAL=$(az identity show -g $RG -n $IDENTITY_NAME --query principalId -o tsv)
export ACR_ID=$(az acr show -g $RG -n $ACR_NAME --query id -o tsv)

az role assignment create --assignee-object-id $IDENTITY_PRINCIPAL \
  --assignee-principal-type ServicePrincipal --role AcrPull --scope $ACR_ID
```

## 4. Container Apps environment

```bash
az containerapp env create -g $RG -n $ENV_NAME -l $LOCATION
```

This also creates a Log Analytics workspace for the app's logs.

## 5. The app, with secrets, scale-to-zero and a placeholder image

The real image does not exist yet, so the app starts from Microsoft's sample image and the pipeline replaces it.
The two secrets are read from your local `.env` straight into shell variables (never printed, never typed):

```bash
export AZURE_OPENAI_API_KEY=$(uv run python -c "from dotenv import dotenv_values; print(dotenv_values('.env')['AZURE_OPENAI_API_KEY'])")
export AZURE_OPENAI_ENDPOINT=$(uv run python -c "from dotenv import dotenv_values; print(dotenv_values('.env')['AZURE_OPENAI_ENDPOINT'])")
export AZURE_OPENAI_DEPLOYMENT=$(uv run python -c "from dotenv import dotenv_values; print(dotenv_values('.env')['AZURE_OPENAI_DEPLOYMENT'])")

# A NEW random key for the public API (different from the Azure OpenAI key). Saved to a git-ignored file.
export API_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
printf 'DEMO_API_KEY=%s\n' "$API_KEY" >> .env.demo

az containerapp create -g $RG -n $APP_NAME --environment $ENV_NAME \
  --image mcr.microsoft.com/k8se/quickstart:latest \
  --ingress external --target-port 80 \
  --cpu 1.0 --memory 2.0Gi --min-replicas 0 --max-replicas 1 \
  --user-assigned $IDENTITY_ID \
  --secrets api-key="$API_KEY" azure-openai-key="$AZURE_OPENAI_API_KEY" \
  --env-vars API_KEY=secretref:api-key AZURE_OPENAI_API_KEY=secretref:azure-openai-key \
             AZURE_OPENAI_ENDPOINT="$AZURE_OPENAI_ENDPOINT" AZURE_OPENAI_DEPLOYMENT="$AZURE_OPENAI_DEPLOYMENT"

# let the app pull from your registry with the identity from step 3
az containerapp registry set -g $RG -n $APP_NAME --server $ACR_NAME.azurecr.io --identity $IDENTITY_ID

export APP_URL=https://$(az containerapp show -g $RG -n $APP_NAME --query properties.configuration.ingress.fqdn -o tsv)
echo "$APP_URL"
printf 'DEMO_URL=%s\n' "$APP_URL" >> .env.demo
```

- `--min-replicas 0 --max-replicas 1`: scale to zero when idle, never more than one copy.
- `secrets` hold the two secret values inside Azure; `secretref:` points the environment variables at them. The
  key is not in the image, not in GitHub and not in any file in this repository.
- 1 vCPU and 2 GiB are enough for the API (model, policy index and agent), with headroom.

## 6. Let GitHub deploy without a password (OIDC)

OIDC means GitHub proves its identity with a short-lived token for *this repository and branch only*, so there is
no stored Azure password to leak.

```bash
export SUBSCRIPTION_ID=$(az account show --query id -o tsv)
export TENANT_ID=$(az account show --query tenantId -o tsv)
export RG_ID=$(az group show -n $RG --query id -o tsv)

export GH_APP_ID=$(az ad app create --display-name fmcg-copilot-github-deploy --query appId -o tsv)
az ad sp create --id $GH_APP_ID

az ad app federated-credential create --id $GH_APP_ID --parameters '{
  "name": "github-main",
  "issuer": "https://token.actions.githubusercontent.com",
  "subject": "repo:'"$GH_REPO"':ref:refs/heads/main",
  "audiences": ["api://AzureADTokenExchange"]
}'

sleep 30    # let the new service principal propagate
az role assignment create --assignee $GH_APP_ID --role Contributor --scope $RG_ID
az role assignment create --assignee $GH_APP_ID --role AcrPush --scope $ACR_ID

echo "AZURE_CLIENT_ID=$GH_APP_ID"
echo "AZURE_TENANT_ID=$TENANT_ID"
echo "AZURE_SUBSCRIPTION_ID=$SUBSCRIPTION_ID"
```

Only pushes to `main` of `$GH_REPO` can log in as this identity, and it can only touch this resource group.
If `az role assignment create` says it cannot find the principal, wait a minute and run it again.

**If your tenant blocks creating app registrations** ("Insufficient privileges"), use a service principal secret
instead: `az ad sp create-for-rbac --name fmcg-copilot-github-deploy --role Contributor --scopes $RG_ID --json-auth`,
add the `AcrPush` role as above, save the JSON as a GitHub secret `AZURE_CREDENTIALS`, and in
`.github/workflows/ci.yml` replace the three `client-id/tenant-id/subscription-id` lines of the
`azure/login@v2` step with `creds: ${{ secrets.AZURE_CREDENTIALS }}` (and drop `id-token: write`).

## 7. GitHub: secrets and variables

Repository **Settings > Secrets and variables > Actions**.

| Kind | Name | Value |
|---|---|---|
| Secret | `AZURE_CLIENT_ID` | the `AZURE_CLIENT_ID` printed in step 6 |
| Secret | `AZURE_TENANT_ID` | the `AZURE_TENANT_ID` printed in step 6 |
| Secret | `AZURE_SUBSCRIPTION_ID` | the `AZURE_SUBSCRIPTION_ID` printed in step 6 |
| Variable | `ACR_NAME` | your `$ACR_NAME` from step 1 |
| Variable | `AZURE_RESOURCE_GROUP` | `fmcg-copilot-rg` |
| Variable | `CONTAINER_APP_NAME` | `fmcg-copilot-api` |

No Azure OpenAI key and no `API_KEY` go into GitHub: they exist only as Container App secrets (step 5).
(The three IDs are not really secret, but keeping them as secrets is the usual convention.)

With the GitHub CLI instead of the web page:

```bash
gh secret set AZURE_CLIENT_ID --body "$GH_APP_ID"
gh secret set AZURE_TENANT_ID --body "$TENANT_ID"
gh secret set AZURE_SUBSCRIPTION_ID --body "$SUBSCRIPTION_ID"
gh variable set ACR_NAME --body "$ACR_NAME"
gh variable set AZURE_RESOURCE_GROUP --body "$RG"
gh variable set CONTAINER_APP_NAME --body "$APP_NAME"
```

## 8. Deploy

```bash
git push origin main        # or: Actions tab > CI/CD > Run workflow
```

The workflow lints, tests, builds the image, smoke-tests it, pushes it to your registry, rolls out a new
revision and checks `https://<your-app>/health`. The first deploy takes several minutes (image build, then a cold
start). Look at the run's summary for the URL, or: `echo $APP_URL`.

## 9. Try it

```bash
uv run python scripts/demo_remote.py      # reads DEMO_URL and DEMO_API_KEY from .env.demo
```

or by hand (the first call after idle is a cold start of about half a minute):

```bash
curl $APP_URL/health
curl -X POST $APP_URL/ask -H "X-API-Key: $API_KEY" -H 'content-type: application/json' \
  -d '{"question": "Can I run a 30% discount, and how far ahead must I request a promotion?"}'
```

Watch the logs (JSON, with latency and cost per request):

```bash
az containerapp logs show -g $RG -n $APP_NAME --follow
```

## 10. Tear everything down (when you are finished)

Delete these individually; do **not** delete the resource group if anything else lives in it.

```bash
az containerapp delete -g $RG -n $APP_NAME --yes
az containerapp env delete -g $RG -n $ENV_NAME --yes
az acr delete -g $RG -n $ACR_NAME --yes
az identity delete -g $RG -n $IDENTITY_NAME
az ad app delete --id $GH_APP_ID
az monitor log-analytics workspace list -g $RG -o table     # delete the workspace created in step 4, if listed
```

## Operating notes

- **Rotate the API key**: `az containerapp secret set -g $RG -n $APP_NAME --secrets api-key="<new key>"`, then
  `az containerapp revision restart ...` (or just push a commit). Several keys can be valid at once:
  `api-key="<new>,<old>"`.
- **Cold starts**: with min replicas 0 the first request after idle waits for the container to start (image
  pull, loading the model). To avoid it for a live demo, call `/health` a minute beforehand, or set
  `--min-replicas 1` for the day (it then costs compute all day).
- **Apple Silicon**: a Docker image built on an M-series Mac is `arm64`; Azure runs `amd64`. Always let GitHub
  Actions build the deployable image. If you must push by hand, build with `--platform linux/amd64`.
- **Image size and ACR**: every deploy pushes a new tag; Basic has 10 GiB. Delete old tags now and then:
  `az acr repository delete -n $ACR_NAME --image fmcg-copilot:<old-sha> --yes`.
- **Not in the pipeline on purpose**: the live-LLM tests and the evals (they cost tokens and need the Azure
  OpenAI key, which must not be in GitHub). Run `uv run python -m evals.run_evals` locally before releases.
