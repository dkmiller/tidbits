interface Person {
    firstName: string;
    lastName: string;
}

export function greeter(person: Person) {
  return "~Hello, " + person.firstName + " " + person.lastName + "!";
}

// let user = { firstName: "Dan", lastName: "Miller"};

// document.body.textContent = greeter(user);
