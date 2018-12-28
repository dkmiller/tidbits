/** Simple "hello world" application in Scala.
  */
object Hello extends App {
    println("Hello, world from Scala!")

    println(s"Your arguments are: ${args.mkString(",")}")
}
