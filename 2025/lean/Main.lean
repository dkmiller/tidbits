import Src.Basic

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
