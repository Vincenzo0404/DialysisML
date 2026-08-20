from dialysisml.adapters.FFNNAdapter import FFNNAdapter
from dialysisml.adapters.ModelAdapter import ModelAdapter
from dialysisml.modelconf import BaseModelConfig, FFNNConfig


class AdapterFactory:
    # 1. IL REGISTRO: Mappa direttamente la Configurazione al suo Adapter
    _registry = {
        FFNNConfig: FFNNAdapter,
    }

    @classmethod
    def register_model(cls, config_class, adapter_class):
        """Permette di registrare nuovi modelli dall'esterno senza toccare questo file!"""
        cls._registry[config_class] = adapter_class

    @classmethod
    def create_model_adapter(cls, config: BaseModelConfig) -> ModelAdapter:
        """
        Returns an instance of the appropriate model adapter based on the provided configuration.
        """
        # type(config) restituisce esattamente la classe (es. FFNNConfig)
        adapter_class = cls._registry.get(type(config))

        if adapter_class is None:
            raise ValueError(
                f"Configurazione non supportata o non registrata: {type(config)}"
            )

        # Istanzia e restituisce l'adapter passandogli la configurazione
        return adapter_class(config)
