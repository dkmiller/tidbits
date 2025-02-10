# Workspaces

Kubernetes-based remote IDEs "as a service".

```bash
pip install -r requirements.txt

ptah deploy

kubectl logs $(kubectl get pods -o json --selector=app=workspaces -o 'jsonpath={.items[0].metadata.name}')

curl -H "content-type: application/json" -d '{"id": "wksp-3", "variant": "jupyter", "port": 9000}' localhost:8000/workspaces/

curl localhost:8000/workspaces/

# /healthz :: VS Code, /api :: JupyterLab
curl -v localhost:8002/workspaces/wksp-3/9000/

curl -X DELETE localhost:8000/workspaces/wksp-3


# "Hot reload" for the OpenResty proxy.
ptah ssh proxy

# TODO: do that on file change? ( https://superuser.com/a/181543 )
openresty -s reload
```

## Links

- https://fastapi.tiangolo.com/tutorial/sql-databases/
- https://sqlmodel.tiangolo.com/tutorial/fastapi/
- https://github.com/dkmiller/tidbits/tree/f88be89e407cd45bdc7f1e5b5382576e7c5abfc1/2023/kubernetes/examples/e2e/images/api
- https://serverfault.com/a/708779
- https://kubernetes.io/docs/tasks/access-application-cluster/port-forward-access-application-cluster/

## Lessons

- :x: [Don't use SQLModel: use SQLAlchemy instead](https://www.reddit.com/r/FastAPI/comments/1ajdgfj/comment/kp0q03z/)
