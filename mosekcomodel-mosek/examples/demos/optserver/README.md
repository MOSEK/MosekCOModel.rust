This demo demonstrates how to write a different backend for the `ModelAPI`.

It is a simple model object that only supports linear and ranged variables and
constraints and offloads optimization to an OptServer instance.

The example defines a `ModelOptserver` object that implements `mosekcomodel::BaseModelTrait`.
