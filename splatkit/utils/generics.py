from typing import Any, Type, get_args, get_origin

from typing import Any, Type, get_args, get_origin


# def extract_generics(
#     instance: Any,
#     *,
#     owner_name: str | None = None,
# ) -> tuple[type, ...]:
#     """
#     Extract concrete generic parameters for `base` from either:

#     1. Subclass specialization:
#            class X(Base[T]): ...
#     2. Instance specialization:
#            Base[T](...)

#     Subclass specialization takes precedence over instance specialization.

#     Args:
#         instance:
#             The object to inspect.

#         base:
#             The generic base class whose parameters should be extracted.

#         owner_name:
#             Optional human-readable name used in error messages.

#     Returns:
#         Tuple of concrete generic types.

#     Raises:
#         TypeError:
#             If no concrete generic parameters can be inferred.
#     """
#     cls = type(instance)

#     # 1. Try subclass-level generics first
#     try:
#         for b in getattr(cls, "__orig_bases__", ()):
#             if get_origin(b) is base:
#                 return get_args(b)
#     except Exception:
#         pass

#     # 2. Fall back to instance-level generics
#     orig = getattr(instance, "__orig_class__", None)
#     if orig is not None:
#         return get_args(orig)

#     name = owner_name or cls.__name__
#     base_name = base.__name__

#     raise TypeError(
#         f"Failed to infer generic parameters for {name}.\n\n"
#         f"Expected one of:\n"
#         f"  - class X({base_name}[T, ...]): ...\n"
#         f"  - {cls.__name__}[T, ...](...)\n"
#     )


def extrace_instance_generics(
    instance: Any,
) -> tuple[type, ...]:
    """
    Ensure an instance was created from a parameterized generic class and
    return its concrete generic arguments.

    This is required for runtime generic introspection, because Python
    erases generic type information unless the class is instantiated as:

        MyClass[ConcreteType, ...](...)

    If the instance was created without generic parameters, `__orig_class__`
    will not exist and runtime type inspection is impossible.

    Args:
        instance:
            The object whose generic parameters are required.

        owner_name:
            Optional human-readable name used in error messages
            (e.g. "renderer", "trainer").

    Returns:
        A tuple of concrete types used to parameterize the instance.

    Raises:
        TypeError:
            If the instance was not created from a parameterized generic class.
    """
    orig = getattr(instance, "__orig_class__", None)
    if orig is None:
        cls_name = instance.__class__.__name__

        raise TypeError(
            f"Failed to infer generic type parameters for {cls_name} "
            f"({cls_name}).\n\n"
            f"Please instantiate it with concrete generic arguments.\n\n"
            f"Example:\n"
            f"    {cls_name}[MyType, ...](...)"
        )

    return get_args(orig)


def extract_subclass_generics(
    cls: Type,
    base: Type,
) -> tuple[type, ...]:
    """
    Return generic parameters used when subclassing `base`, e.g.:

        class X(Base[A]): ...

    Returns None if `cls` does not specialize `base`.
    """
    for b in getattr(cls, "__orig_bases__", ()):
        origin = get_origin(b)
        if origin is None:
            continue

        if issubclass(origin, base):
            return get_args(b)

    raise TypeError(
        f"Class {cls.__name__} does not specialize {base.__name__}"
    )