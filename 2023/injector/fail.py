# None of this works

from typing import Generic, TypeVar, get_args

from injector import Injector, Module, provider

T = TypeVar("T")
S = TypeVar("S")


# @inject
class Requestable(Generic[T]):
    def method(self) -> T:
        _t = type(self)
        print(f"type == {_t}")
        bases = _t.__orig_bases__
        for index, base in enumerate(bases):
            print(f"Base {index} --> {base}")
            print(f"Args -> {get_args(base)}")
        # for base in
        # print(get_args(type(self).__orig_bases__[1]))

    @classmethod
    def class_method(cls):
        print(f"Class method args -> {get_args(cls)}")


class R2(T):
    pass


class Builder(Module):
    @provider
    def build_requestable(self) -> Requestable[str]:
        print(S)
        pass

    @provider
    def build_r2(self) -> R2:
        print("r2")
        pass


Requestable[str]().method()
Requestable[str].class_method()
from typing import List

print(get_args(List[str]))
print(get_args(Requestable[str]))


injector = Injector([Builder()], auto_bind=True)

injector.get(R2)


injector.get(Requestable[str])
