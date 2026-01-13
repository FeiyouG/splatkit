from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar, Any
import pickle

T = TypeVar("T")

@dataclass
class LazyInit(Generic[T]):
    """
    Lazy initializer for deferred object construction.
    
    Note:
    - The factory must be picklable (module-level function or class).
    - The object is immutatble, and each call to `init` will return a new instance.

    Example:
        >>> lazy_dataset = LazyInit(ColmapDataset, colmap_dir="data/garden/sparse/0")
        >>> dataset = lazy_dataset.new_instance()  # Constructs a new ColmapDataset instance
        >>> dataset = lazy_dataset.singleton_instance()  # Returns the singleton instance
    """
    _factory: Callable[..., T]
    _kwargs: dict[str, Any]
    
    def __init__(self, factory: Callable[..., T], **kwargs):
        """
        Create a lazy wrapper.
        
        Args:
            factory: A picklable callable (module-level function or class)
            **kwargs: Arguments to pass to factory when called
        """
        self._factory = factory
        self._kwargs = kwargs

        self.__validate__()
    
    def __validate__(self):
        """Validate the lazy initializer."""
        pass
    
    def new_instance(self) -> T:
        """Initialize the instance."""
        return self._factory(**self._kwargs)
    
    def singleton_instance(self) -> T:
        """Get the singleton instance."""
        if not hasattr(self, '_singleton_instance'):
            self._singleton_instance = self._factory(**self._kwargs)
        return self._singleton_instance
    
    def __getstate__(self):
        """
        Custom pickle support.
        
        Handles both regular functions and lambdas.
        """
        # Try to pickle normally first
        try:
            # Test if factory is picklable
            pickle.dumps(self._factory)
            # If we got here, it's picklable
            return {
                '_factory': self._factory,
                '_kwargs': self._kwargs,
                '_uses_dill': False,
            }
        except (pickle.PicklingError, AttributeError, TypeError):
            # Not picklable with standard pickle (e.g., lambda)
            # Fall back to dill
            try:
                import dill
                return {
                    '_factory': dill.dumps(self._factory),
                    '_kwargs': self._kwargs,
                    '_uses_dill': True,
                }
            except ImportError:
                raise ImportError(
                    "Lazy wrapper with lambda/local functions requires 'dill' package. "
                    "Install with: pip install dill"
                )
    
    def __setstate__(self, state):
        """Custom unpickle support."""
        if state.get('_uses_dill', False):
            # Deserialize with dill
            import dill
            self._factory = dill.loads(state['_factory'])
        else:
            # Standard pickle
            self._factory = state['_factory']
        
        self._kwargs = state['_kwargs']
    
    def __repr__(self):
        return f"LazyInit(factory={self._factory.__name__}, kwargs={self._kwargs})"