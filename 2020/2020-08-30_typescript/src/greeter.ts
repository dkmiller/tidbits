interface Person {
    firstName: string;
    lastName: string;
}

export function greeter(person: Person) {
  return "Hello, " + person.firstName + " " + person.lastName + "!";
}

export function showGreeting(divName: string, firstName: string, lastName: string) {
  const elt = document.getElementById(divName);
  let innerText = greeter({ firstName: firstName, lastName: lastName });
  console.log(innerText);
  elt.innerText = innerText;
}

// let user = { firstName: "Dan", lastName: "Miller"};

// document.body.textContent = greeter(user);
