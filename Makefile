# Namespace used for all deployments
NAMESPACE=rag-arxiv-app

# All kustomize directories
KUSTOMIZE_DIRS := k8s/redis k8s/index-service k8s/vector-db-service k8s/query-service k8s/prometheus k8s/grafana
# KUSTOMIZE_DIRS := k8s/redis k8s/vector-db-service

# Default target: deploy everything
.PHONY: all
all: deploy

# Deploy all services
.PHONY: deploy
deploy:
	@echo "Deploying all services..."
	@kubectl create namespace $(NAMESPACE) --dry-run=client -o yaml | kubectl apply -f -
	@for dir in $(KUSTOMIZE_DIRS); do \
		echo "Applying $$dir..."; \
		kubectl apply -k $$dir; \
	done
	@echo "All services deployed."

# Delete all services but keep PVCs
.PHONY: clean
clean:
	@echo "Deleting all deployments, services, configmaps, secrets..."
	@for dir in $(KUSTOMIZE_DIRS); do \
		kubectl delete -k $$dir --ignore-not-found; \
	done
	@echo "All pods, deployments, services, configmaps, secrets deleted (PVCs remain)."

# Delete everything including PVCs
.PHONY: clean-all
clean-all:
	@echo "Deleting all resources including PVCs..."
	@for dir in $(KUSTOMIZE_DIRS); do \
		kubectl delete -k $$dir --ignore-not-found; \
	done
	@echo "Deleting PVCs..."
	@kubectl delete pvc --all -n $(NAMESPACE) --ignore-not-found
	@echo "All resources deleted."

# Re-deploy from clean state (delete everything + deploy)
.PHONY: redeploy
redeploy: clean-all deploy
	@echo "Re-deployed all services from scratch."

# Port-forward shortcuts
# .PHONY: port-forward-index port-forward-query port-forward-vector-db port-forward-prometheus port-forward-grafana
# port-forward-index:
# 	kubectl port-forward svc/index-service 9000:80 -n $(NAMESPACE)

# port-forward-query:
# 	kubectl port-forward svc/query-service 9001:80 -n $(NAMESPACE)

# port-forward-vector-db:
# 	kubectl port-forward svc/vector-db-service 8000:80 -n $(NAMESPACE)

# port-forward-prometheus:
# 	kubectl port-forward svc/prometheus 9090:9090 -n $(NAMESPACE)

# port-forward-grafana:
# 	kubectl port-forward svc/grafana 3000:3000 -n $(NAMESPACE)
.PHONY: port-forward-all
port-forward-all:
	kubectl port-forward svc/index-service 9000:80 -n $(NAMESPACE) & \
	kubectl port-forward svc/query-service 9001:80 -n $(NAMESPACE) & \
	kubectl port-forward svc/vector-db-service 8000:80 -n $(NAMESPACE) & \
	kubectl port-forward svc/prometheus 9090:9090 -n $(NAMESPACE) & \
	kubectl port-forward svc/grafana 3000:3000 -n $(NAMESPACE) & \
	wait