/** Simple "hello world" application in Scala.
  *
  * It is NOT intended to be a part of the Scala project under the "src" directory.
  */
object Hello extends App {
    println("Hello, world from Scala!")

    println(s"Your arguments are: ${args.mkString(",")}")
}
