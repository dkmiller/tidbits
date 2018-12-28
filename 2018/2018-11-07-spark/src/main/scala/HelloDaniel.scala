package HelloDaniel

/** Simple class for learning about Scala's features.
  */
object Main extends App {
    val p = Person("Daniel Miller")

    println(s"Hi from Scala (${p.name})!")
}

case class Person(var name: String)