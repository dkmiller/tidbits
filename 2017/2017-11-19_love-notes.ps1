<#
This script reads a CSV file tracking my daily work on 'Love Notes'
for Ivy, and prints a summary of how many days I have to spare at my
current rate.

Sample use:

.\2017-11-19_love-notes.ps1
#>

Param(
  $CsvFile = '~\Dropbox\diary\love-notes.csv'
)

$data = Import-Csv $CsvFile

$ValentinesDay = [datetime]'2018-02-14'

# How many more days worth of 'love notes' need to be written after this study started?
$NotesRequired = $ValentinesDay - [datetime]'2017-08-08'

# Average number of 'love notes' written daily.
$AverageWork = $data | Where-Object {[datetime]$_.Date -le (get-date)} | Measure-Object Num -Average

# How many days left until completion (at current rate)?
$DaysLeft = [int][Math]::Ceiling($NotesRequired.Days / $AverageWork.Average)

$DaysToSpare = ($ValentinesDay - (Get-Date)).Days - $DaysLeft

Write-Output "At current rate, you have $DaysToSpare days to spare."
