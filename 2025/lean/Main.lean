import Src.Basic

def triple :=
  λ (n: Nat) => 3 * n

-- A kind of generics, nice!
def multiplyFuncs (α β: Type) (g: α → Nat) (h: β → Nat) :=
  λ (a: α) (b: β) => g a * h b

def main : IO Unit :=
  IO.println s!"Hello, {hello}!"

  #check Nat → Nat
  #check Nat × Nat
  def α: Type := Nat
  #check α
  def b: Nat := 1
  def a: α := b
  #check a
  #check Type
  #check List
  def f: (Nat → Nat) :=  λ (x: Nat) => x * x
  #check f
  #check λ (u: Nat → Nat)(v: Nat → Nat)(x: Nat) => u x * v x
  #eval triple 7
