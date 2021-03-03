# Scraping web pages

Learn about how to scrape web pages using Python, summarize those stats using
PowerShell.

To run, execute:

```powershell
$Domain = "yourdomain"
$env:START_URL = "https://$($Domain).com/"

# Scrape pages.
scrapy runspider scraper.py -o pages.jl

# Get Wordpress "page view" stats per page (get nonce from dev tools).
.\Get-Stats.ps1 -File .\pages4-next.jl -Nonce "NONCE" -Domain $Domain

# Get top pages.
Get-Content .\pages.jl-stats.csv | ConvertFrom-Csv | Sort-Object -Property @{Expression={$_.VoteCount * $_.AvgRating}; Descending = $True} | select -First 20
```

## Links

- [How To Crawl A Web Page with Scrapy and Python 3](https://www.digitalocean.com/community/tutorials/how-to-crawl-a-web-page-with-scrapy-and-python-3)
- [Is there a way to get the XPath in Google Chrome?](https://stackoverflow.com/a/46599584)
- [What is the easiest way to get current GMT time in Unix timestamp format?](https://stackoverflow.com/a/49362936)
- [`Add-Member`](https://docs.microsoft.com/en-us/powershell/module/microsoft.powershell.utility/add-member?view=powershell-7.1)
- [How to append data into a CSV file using PowerShell?](https://www.tutorialspoint.com/how-to-append-data-into-a-csv-file-using-powershell)
- [`Sort-Object` &gt; Example 6: Sort text files by time span](https://docs.microsoft.com/en-us/powershell/module/microsoft.powershell.utility/sort-object?view=powershell-7.1#example-6--sort-text-files-by-time-span)
