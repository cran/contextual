library(here)
library(contextual)
library(data.table)
library(DBI)
library(MonetDB.R)
library(doParallel)
library(httr)

# Config -----------------------------------------------------------------------------------------------------

simulations             <- 1
horizon                 <- 37450000
buffer_size             <- 1000
sparsity_vector         <- c(0.00,0.70,0.80,0.90,0.95,0.99)
worker_max              <- 8

monetdb_host            <- "monetdb_ip"
monetdb_dbname          <- "yahoo"
monetdb_user            <- "monetdb"
monetdb_password        <- "monetdb"

setwd(here::here("demo", "replication_li_2010"))

# Setup ------------------------------------------------------------------------------------------------------

doParallel::stopImplicitCluster()

source("./demo_yahoo_classes/yahoo_bandit.R", encoding="utf-8")
source("./demo_yahoo_classes/yahoo_policy_epsilon_greedy.R", encoding="utf-8")
source("./demo_yahoo_classes/yahoo_policy_epsilon_greedy_seg.R", encoding="utf-8")
source("./demo_yahoo_classes/yahoo_policy_ucb1_alpha.R", encoding="utf-8")
source("./demo_yahoo_classes/yahoo_policy_ucb1_alpha_seg.R", encoding="utf-8")
source("./demo_yahoo_classes/yahoo_policy_linucb_disjoint.R", encoding="utf-8")
source("./demo_yahoo_classes/yahoo_policy_linucb_hybrid.R", encoding="utf-8")
source("./demo_yahoo_classes/yahoo_policy_random.R", encoding="utf-8")

# Connect to DB ----------------------------------------------------------------------------------------------

con <- DBI::dbConnect(MonetDB.R(), host=monetdb_host, dbname=monetdb_dbname,
                      user=monetdb_user, password=monetdb_password)

message(paste0("MonetDB: connection to '",dbListTables(con),"' database succesful!"))

arm_lookup_table <- as.matrix(DBI::dbGetQuery(con, "SELECT DISTINCT article_id FROM yahoo"))
class(arm_lookup_table) <- "integer"
arm_lookup_table <- rev(as.vector(arm_lookup_table))

# YahooBandit Loop ---------------------------------------------------------------------------------------

for (sparsity in sparsity_vector) {

  save_file_name          <- paste0("Yahoo_T_",horizon,"_sparse_",sparsity,".RData")

  bandit <- YahooBandit$new(k = 217L, unique = c(1:6), shared = c(7:12),
                            arm_lookup = arm_lookup_table, host = monetdb_host,
                            dbname = monetdb_dbname, user = monetdb_user,
                            password = monetdb_password, buffer_size = buffer_size)

  agents <-
    list (Agent$new(YahooLinUCBDisjointPolicy$new(0.2),       bandit, name = "LinUCB Dis",  sparse = sparsity),
          Agent$new(YahooLinUCBHybridPolicy$new(0.2),         bandit, name = "LinUCB Hyb",  sparse = sparsity),
          Agent$new(YahooEpsilonGreedyPolicy$new(0.3),        bandit, name = "EGreedy",     sparse = sparsity),
          Agent$new(YahooEpsilonGreedySegPolicy$new(0.3),     bandit, name = "EGreedySeg",  sparse = sparsity),
          Agent$new(YahooUCB1AlphaPolicy$new(0.4),            bandit, name = "UCB1",        sparse = sparsity),
          Agent$new(YahooUCB1AlphaSegPolicy$new(0.4),         bandit, name = "UCB1Seg",     sparse = sparsity),
          Agent$new(YahooRandomPolicy$new(),                  bandit, name = "Random"))

  simulation <- Simulator$new(
    agents,
    simulations = simulations,
    horizon = horizon,
    do_parallel = TRUE,
    worker_max = worker_max,
    progress_file = TRUE,
    include_packages = c("MonetDB.R"))

  history  <- simulation$run()

  history$save(save_file_name)

  history$clear_data_table()
  gc()

}

# Take a look at the results ---------------------------------------------------------------------------------

print(history$meta$sim_total_duration)

plot(history, regret = FALSE, rate = TRUE, type = "cumulative", ylim = c(0.035,0.07))

dbDisconnect(con)
