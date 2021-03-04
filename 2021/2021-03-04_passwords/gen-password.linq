<Query Kind="Statements">
  <NuGetReference>Bogus</NuGetReference>
  <Namespace>Bogus</Namespace>
</Query>

var f = new Faker();

var possibilities = new List<char>();

var upperCaseCharacters = Enumerable.Range(65, 26).Select(i => (char)i);
possibilities.AddRange(upperCaseCharacters);

var lowerCaseCharacters = Enumerable.Range(97, 26).Select(i => (char)i);
possibilities.AddRange(lowerCaseCharacters);

var numbers = Enumerable.Range(48, 10).Select(i => (char)i);
possibilities.AddRange(numbers);

var specialCharacters = "!@#$%^&*()_+=[{]};:<>|./?,-".ToList();
possibilities.AddRange(specialCharacters);

//possibilities.Dump();

var password = "";

for (var i = 0; i < 20; ++i)
{
	password += f.PickRandom(possibilities);
}

password.Dump();
