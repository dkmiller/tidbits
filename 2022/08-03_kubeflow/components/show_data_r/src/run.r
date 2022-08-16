library("optparse")

option_list <- list( 
    make_option("--input")
)
opt <- parse_args(OptionParser(option_list=option_list))

dir(opt$input)
