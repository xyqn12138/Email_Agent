"""New LangGraph Agent."""

__all__ = ["graph"]


def __getattr__(name: str):
    if name == "graph":
        from agent.graph import graph

        return graph
    raise AttributeError(f"module 'agent' has no attribute {name!r}")
