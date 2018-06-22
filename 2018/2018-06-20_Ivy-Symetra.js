// Code to compute number of work days between two dates.

var startDate = new Date("2018-01-01");
var endDate = new Date("2018-01-11");

var currentDate = startDate;

var numWorkDays = 0;

while (currentDate <= endDate)
{
    var currentDay = currentDate.getDay();
    
    if (currentDay !== 0 && currentDay !== 6)
    {
        numWorkDays++;
    }

    currentDate.setDate(currentDate.getDate() + 1);
}

alert(numWorkDays + " ");

