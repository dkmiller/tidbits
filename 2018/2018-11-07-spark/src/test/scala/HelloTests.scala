package HelloDaniel

import org.scalatest.FunSuite

class HelloTests extends FunSuite {

    test("Constructor works") {
        val name = "xsef9us0dfu"
        val p = Person(name)
        assert(p.name === name)
    }
}
