#' @export
Plot <- R6::R6Class(
  "Plot",
  public = list(
    history = NULL,

    cumulative = function(history,

                          regret             = TRUE,
                          disp               = NULL,
                          plot_only_disp     = FALSE,
                          rate               = FALSE,
                          interval           = 1,
                          traces             = FALSE,
                          traces_max         = 100,
                          traces_alpha       = 0.3,
                          smooth             = FALSE,
                          no_par             = FALSE,
                          xlim               = NULL,
                          ylim               = NULL,
                          xlab               = NULL,
                          ylab               = NULL,
                          legend             = TRUE,
                          log                = "",
                          use_colors         = TRUE,
                          color_step         = 1,
                          lty_step           = 1,
                          lwd                = 2,
                          legend_labels      = NULL,
                          legend_border      = NULL,
                          legend_position    = "topleft",
                          legend_title       = NULL,
                          limit_agents       = NULL,
                          limit_context      = NULL,
                          trunc_over_agents  = TRUE,
                          trunc_per_agent    = TRUE) {

      self$history       <- history

      if (regret) {
        if (rate) {
          ylab_title     <- "Cumulative regret rate"
          line_data_name <- "cum_regret_rate"
          disp_data_name <- "cum_regret_rate_none"
        } else {
          ylab_title     <- "Cumulative regret"
          line_data_name <- "cum_regret"
          disp_data_name <- "cum_regret_none"
        }
      } else {
        if (rate) {
          ylab_title     <- "Cumulative reward rate"
          line_data_name <- "cum_reward_rate"
          disp_data_name <- "cum_reward_rate_none"
        } else {
          ylab_title     <- "Cumulative reward"
          line_data_name <- "cum_reward"
          disp_data_name <- "cum_reward_none"
        }
      }

      private$do_plot(
        line_data_name      = line_data_name,
        disp_data_name      = disp_data_name,
        ylab_title          = ylab_title,
        use_colors          = use_colors,
        log                 = log,
        legend              = legend,
        disp                = disp,
        plot_only_disp      = plot_only_disp,
        no_par              = no_par,
        interval            = interval,
        color_step          = color_step,
        lty_step            = lty_step,
        lwd                 = lwd,
        xlim                = xlim,
        ylim                = ylim,
        xlab                = xlab,
        ylab                = ylab,
        legend_labels       = legend_labels,
        legend_border       = legend_border,
        legend_position     = legend_position,
        legend_title        = legend_title,
        limit_agents        = limit_agents,
        limit_context       = limit_context,
        traces              = traces,
        traces_max          = traces_max,
        traces_alpha        = traces_alpha,
        smooth              = smooth,
        rate                = rate,
        trunc_over_agents   = trunc_over_agents,
        trunc_per_agent     = trunc_per_agent
      )

      invisible(recordPlot())
    },

    optimal = function(history,
                       disp               = NULL,
                       plot_only_disp     = FALSE,
                       rate               = FALSE,
                       interval           = 1,
                       traces             = FALSE,
                       traces_max         = 100,
                       traces_alpha       = 0.3,
                       smooth             = FALSE,
                       no_par             = FALSE,
                       xlim               = NULL,
                       ylim               = NULL,
                       xlab               = NULL,
                       ylab               = NULL,
                       legend             = TRUE,
                       use_colors         = TRUE,
                       log                = "",
                       color_step         = 1,
                       lty_step           = 1,
                       lwd                = 2,
                       legend_labels      = NULL,
                       legend_border      = NULL,
                       legend_position    = "topleft",
                       legend_title       = NULL,
                       limit_agents       = NULL,
                       limit_context      = NULL,
                       trunc_over_agents  = TRUE,
                       trunc_per_agent    = TRUE) {

      self$history <- history

      ylab_title     <- "Optimal action"
      line_data_name <- "optimal"
      disp_data_name   <- "optimal_none"

      private$do_plot(
        line_data_name      = line_data_name,
        disp_data_name      = disp_data_name,
        ylab_title          = ylab_title,
        use_colors          = use_colors,
        log                 = log,
        legend              = legend,
        disp                = disp,
        plot_only_disp      = plot_only_disp,
        no_par              = no_par,
        interval            = interval,
        color_step          = color_step,
        lty_step            = lty_step,
        lwd                 = lwd,
        xlim                = xlim,
        ylim                = ylim,
        legend_labels       = legend_labels,
        legend_border       = legend_border,
        legend_position     = legend_position,
        legend_title        = legend_title,
        limit_agents        = limit_agents,
        limit_context       = limit_context,
        traces              = traces,
        traces_max          = traces_max,
        traces_alpha        = traces_alpha,
        smooth              = smooth,
        trunc_over_agents   = trunc_over_agents,
        trunc_per_agent     = trunc_per_agent
      )

      invisible(recordPlot())
    },


    average = function(history,
                       regret             = TRUE,
                       disp               = NULL,
                       plot_only_disp     = FALSE,
                       rate               = FALSE,
                       interval           = 1,
                       traces             = FALSE,
                       traces_max         = 100,
                       traces_alpha       = 0.3,
                       smooth             = FALSE,
                       no_par             = FALSE,
                       xlim               = NULL,
                       ylim               = NULL,
                       xlab               = NULL,
                       ylab               = NULL,
                       legend             = TRUE,
                       use_colors         = TRUE,
                       log                = "",
                       color_step         = 1,
                       lty_step           = 1,
                       lwd                = 2,
                       cum_average        = FALSE,
                       legend_labels      = NULL,
                       legend_border      = NULL,
                       legend_position    = "topleft",
                       legend_title       = NULL,
                       limit_agents       = NULL,
                       limit_context      = NULL,
                       trunc_over_agents  = TRUE,
                       trunc_per_agent    = TRUE) {
      self$history <- history

      if (regret) {
        ylab_title     <- "Average regret"
        line_data_name <- "regret"
        disp_data_name   <- "regret_none"
      } else {
        ylab_title     <- "Average reward"
        line_data_name <- "reward"
        disp_data_name   <- "reward_none"
      }

      private$do_plot(
        line_data_name      = line_data_name,
        disp_data_name      = disp_data_name,
        ylab_title          = ylab_title,
        use_colors          = use_colors,
        log                 = log,
        legend              = legend,
        disp                = disp,
        plot_only_disp      = plot_only_disp,
        no_par              = no_par,
        interval            = interval,
        color_step          = color_step,
        lty_step            = lty_step,
        lwd                 = lwd,
        xlim                = xlim,
        ylim                = ylim,
        xlab                = xlab,
        ylab                = ylab,
        legend_labels       = legend_labels,
        legend_border       = legend_border,
        legend_position     = legend_position,
        legend_title        = legend_title,
        cum_average         = cum_average,
        limit_agents        = limit_agents,
        limit_context       = limit_context,
        traces              = traces,
        traces_max          = traces_max,
        traces_alpha        = traces_alpha,
        smooth              = smooth,
        rate                = rate,
        trunc_over_agents   = trunc_over_agents,
        trunc_per_agent     = trunc_per_agent
      )

      invisible(recordPlot())
    },

    arms = function(history,

                    no_par             = FALSE,
                    legend             = TRUE,
                    use_colors         = TRUE,
                    log                = "",
                    interval           = 1,
                    xlim               = NULL,
                    ylim               = NULL,
                    xlab               = NULL,
                    ylab               = NULL,
                    legend_labels      = NULL,
                    legend_border      = NULL,
                    legend_position    = "bottomright",
                    legend_title       = NULL,
                    limit_context      = NULL,
                    smooth             = FALSE,
                    trunc_over_agents  = TRUE,
                    limit_agents       = NULL) {

      self$history <- history

      if (!isTRUE(no_par)) {
        dev.hold()
        old.par <- par(no.readonly = TRUE)
        par(mar = c(5, 5, 1, 1))
      }

      if(!is.null(limit_context)) {
        dt <- self$history$get_data_table(
          limit_cols   = c("agent", "t", "choice", "sim", limit_context),
          limit_agents = limit_agents,
          interval     = interval
        )
        dt <- dt[dt[, Reduce(`|`, lapply(.SD, `==`, 1)),.SDcols = limit_context],]
      } else {
        dt <- self$history$get_data_table(
          limit_cols   = c("agent", "t", "choice", "sim"),
          limit_agents = limit_agents,
          interval     = interval
        )
      }

      if(isTRUE(trunc_over_agents))  {
        min_t_sim <- min(dt[,max(t), by = c("agent","sim")]$V1)
        dt <- dt[t<=min_t_sim]
      }

      ylab_title        <- "Arm choice %"
      agent_levels      <- levels(droplevels(dt$agent))

      if (length(agent_levels) > 1) {
        warning(strwrap(
          prefix = " ", initial = "",
          "## Arm percentage plot always plots the results of one agent, either at
          index position one, or the first agent specified in limit_agents."
        ),
        call. = FALSE
        )
      }

      dt                <- dt[agent == agent_levels[1]]

      dt$agent          <- NULL
      data.table::setkey(dt, t, choice)
      data              <- dt[data.table::CJ(t, choice, unique = TRUE), list(arm_count =  .N), by = .EACHI]

      #data              <- dt[, list(arm_count =  .N), by = list(t, choice)]

      max_sim           <- dt[, max(sim)]
      max_t             <- dt[, max(t)]

      arm_levels        <- levels(as.factor(data$choice))
      max_arm           <- length(arm_levels)
      N                 <- dt[,.N,by=c("t")]$N
      N                 <- rep(N,each=max_arm)

      data$arm_count    <- as.double((unlist(data$arm_count, FALSE, FALSE) / N) * 100L)

      eg                <- expand.grid(t = dt[sim == 1]$t, choice = seq(1.0, max_arm, 1))
      data              <- merge(data, eg, all = TRUE)
      # turn NA into 0
      for (j in seq_len(ncol(data)))
        set(data,which(is.na(data[[j]])),j,0)

      data$dataum       <- ave(data$arm_count, data$t, FUN = cumsum)
      data$zero         <- 0.0
      min_ylim          <- 0
      max_ylim          <- 100

      if (isTRUE(smooth)) {
        for (arm in arm_levels) {
          data[data$choice == arm, c("t", "dataum") :=
                 supsmu(data[data$choice == arm]$t, data[data$choice == arm]$dataum, bass = 9)]
        }
      }

      data.table::setorder(data, choice, t)
      plot.new()

      if (!is.null(xlim)) {
        min_xlim <- xlim[1]
        max_xlim <- xlim[2]
      } else {
        min_xlim <- 1
        max_xlim <- data[, max(t)]
      }
      if (!is.null(ylim)) {
        min_ylim <- ylim[1]
        max_ylim <- ylim[2]
      }
      plot.window(
        xlim = c(min_xlim, max_xlim),
        ylim = c(min_ylim, max_ylim)
      )

      if (isTRUE(use_colors)) {
        cl <- private$gg_color_hue(length(arm_levels))
      } else {
        cl <- gray.colors(length(arm_levels))
      }

      color <- 1
      polygon(
        c(data[data$choice == 1]$t, rev(data[data$choice == 1]$t)),
        c(data[data$choice == 1]$dataum, rev(data[data$choice == 1]$zero)),
        col = adjustcolor(cl[color], alpha.f = 0.6),
        border = NA
      )

      color <- 2
      for (arm_nr in c(2:length(arm_levels))) {
        polygon(
          c(data[data$choice == arm_nr]$t, rev(data[data$choice == arm_nr]$t)),
          c(data[data$choice == arm_nr - 1]$dataum, rev(data[data$choice == arm_nr]$dataum)),
          col = adjustcolor(cl[color], alpha.f = 0.6),
          border = NA
        )
        color <- color + 1
      }

      if (is.null(legend_title)) {
        legend_title <- agent_levels[1]
        if(!is.null(limit_context))
          legend_title <- paste(legend_title,limit_context)
      }

      if (is.null(legend_position)) {
        legend_position <- "bottomright"
      }

      if (!is.null(legend_labels)) {
        legend_labels <- legend_labels
      } else {
        legend_labels <- paste("arm", arm_levels, sep = " ")
      }

      axis(1)
      axis(2)
      title(xlab = "Time Step")
      title(ylab = ylab_title)
      box()
      if (legend) {
        legend(
          legend_position,
          NULL,
          legend_labels,
          col = adjustcolor(cl, alpha.f = 0.6),
          title = legend_title,
          pch = 15,
          pt.cex = 1.2,
          bg = "white",
          inset = c(0.08, 0.1)
        )
      }
      if (!isTRUE(no_par)) {
        dev.flush()
        par(old.par)
      }
      invisible(recordPlot())
    }
  ),
  private = list(
    cum_average = function(cx) {
      cx <- c(0,cx)
      cx[(2):length(cx)] - cx[1:(length(cx) - 1)]
    },
    do_plot = function(line_data_name      = line_data_name,
                       disp_data_name      = disp_data_name,
                       disp                = NULL,
                       plot_only_disp      = FALSE,
                       ylab_title          = NULL,
                       use_colors          = FALSE,
                       log                 = "",
                       legend              = TRUE,
                       no_par              = FALSE,
                       xlim                = NULL,
                       ylim                = NULL,
                       xlab                = NULL,
                       ylab                = NULL,
                       interval            = 1,
                       color_step          = 1,
                       lty_step            = 1,
                       lwd                 = 2,
                       legend_labels       = NULL,
                       legend_border       = NULL,
                       legend_position     = "topleft",
                       legend_title        = NULL,
                       limit_agents        = NULL,
                       limit_context       = NULL,
                       traces              = NULL,
                       traces_max          = 100,
                       traces_alpha        = 0.3,
                       cum_average         = FALSE,
                       smooth              = FALSE,
                       rate                = FALSE,
                       trunc_over_agents   = TRUE,
                       trunc_per_agent     = TRUE) {

      cum_flip <- FALSE
      if((line_data_name=="reward" || line_data_name=="regret") && isTRUE(cum_average)) {
        line_data_name <- paste0("cum_",line_data_name)
        cum_flip = TRUE
      }

      if (interval==1 && as.integer(self$history$meta$sim$max_t) > 1850) {
        interval <- ceiling(as.integer(self$history$meta$sim$max_t)/1850) # nocov
        if(isTRUE(cum_average) && isTRUE(cum_flip))  {
           warning(strwrap(
            prefix = " ", initial = "",
            paste0("## As cum_reward was set to TRUE while plotting more than 1850 time steps,
            the reward plot has been smoothed automatically using a window length of ",interval,
            " timesteps.")
          ),
          call. = FALSE
          )
        }
      }

      if (!is.null(disp) && disp %in% c("sd", "var", "ci")) {

        disp_data_name <- gsub("none", disp, disp_data_name)
        data <-
          self$history$get_cumulative_data(
            limit_cols   = c("agent", "t", "sims", line_data_name, disp_data_name),
            limit_agents = limit_agents,
            interval     = interval
          )

      } else {
        disp <- NULL
        data <-
          self$history$get_cumulative_data(
            limit_cols   = c("agent", "t", "sims", line_data_name),
            limit_agents = limit_agents,
            interval     = interval
          )
      }

      agent_levels <- levels(droplevels(data$agent))
      n_agents <- length(agent_levels)

      # turn NA into 0
      for (j in seq_len(ncol(data)))
        data.table::set(data,which(is.na(data[[j]])),j,0)

      if(isTRUE(trunc_per_agent))  {
        data <- data[data$sims == max(data$sims)]
      }

      if(isTRUE(trunc_over_agents))  {
        min_t_sim <- min(data[,max(t), by = c("agent")]$V1)
        data <- data[t<=min_t_sim]
      }

      if (!is.null(xlim)) {
        min_xlim <- xlim[1]
        max_xlim <- xlim[2]
      } else {
        min_xlim <- 1
        max_xlim <- data[, max(t)]
      }

      data.table::setorder(data, agent, t)

      if(cum_flip==TRUE) {
        if (line_data_name == "cum_reward") {
          line_data_name <- "reward"
          for (agent_name in agent_levels) {
            data[data$agent == agent_name,
                 reward := private$cum_average(data[data$agent == agent_name]$cum_reward)/interval]
          }
        } else {
          line_data_name <- "cum_regret"
          for (agent_name in agent_levels) {
            data[data$agent == agent_name,
                 regret := private$cum_average(data[data$agent == agent_name]$cum_regret)/interval]
          }
        }
      }

      if(!is.null(xlim)) data <- data[t>=xlim[1] & t<=xlim[2]]

      if(!is.null(limit_context)) {
        data <- data[data[, Reduce(`|`, lapply(.SD, `==`, 1)),.SDcols = sel],]
      }

      data.table::setorder(data, agent, t)

      if (isTRUE(smooth)) {
        for (agent_name in agent_levels) {
          data[data$agent == agent_name, c("t", line_data_name) :=
                 supsmu(data[data$agent == agent_name]$t, data[data$agent == agent_name][[line_data_name]])]
          if (!is.null(disp)) {
            data[data$agent == agent_name, c("t", disp_data_name) :=
                   supsmu(data[data$agent == agent_name]$t, data[data$agent == agent_name][[disp_data_name]])]
          }
        }
      }

      if (!isTRUE(no_par)) {
        dev.hold()
        old.par <- par(no.readonly = TRUE)
        par(mar = c(5, 5, 1, 1))
      }

      if (!is.null(disp) && !isTRUE(plot_only_disp)) {
        disp_range <- data[[line_data_name]] + outer(data[[disp_data_name]], c(1, -1))
        data     <- cbind(data, disp_range)
        colnames(data)[colnames(data) == "V2"] <- "disp_lower"
        colnames(data)[colnames(data) == "V1"] <- "disp_upper"
      }

      if (isTRUE(plot_only_disp)) {
        if(is.null(disp)) stop("Need to set disp to 'var','sd' or 'ci' when plot_only_disp is TRUE",
                               call. = FALSE)
        line_data_name = disp_data_name
      }

      plot.new()
      cl <- private$gg_color_hue(round(n_agents / color_step))
      cl <- rep(cl, round(color_step))

      if (lty_step > 1) {
        lt <- rep(1:round(lty_step), each = round(n_agents / lty_step))
      } else {
        lt <- rep(1, n_agents)
      }
      if (!isTRUE(use_colors) && lty_step == 1) {
        lty_step <- n_agents
        lt <- rep(1:round(lty_step), each = round(n_agents / lty_step))
      }

      if (!is.null(disp) && !isTRUE(plot_only_disp) &&
          !is.na(data[, min(disp_lower)]) && !is.na(data[, min(disp_upper)])) {
        min_ylim <- data[, min(disp_lower)]
        max_ylim <- data[, max(disp_upper)]
      } else {
        min_ylim <- data[, min(data[[line_data_name]])]
        max_ylim <- data[, max(data[[line_data_name]])]
      }


      if (!is.null(ylim)) {
        min_ylim <- ylim[1]
        max_ylim <- ylim[2]
      }
      plot.window(
        xlim = c(min_xlim, max_xlim),
        ylim = c(min_ylim, max_ylim),
        log = log
      )

      if (isTRUE(traces) && !isTRUE(plot_only_disp)) {
        dt <- self$history$get_data_table(limit_agents = limit_agents, interval = interval)
        data.table::setorder(dt, agent, sim, t)
        for (agent_name in agent_levels) {
          agent_sims <- unique(dt[dt$agent == agent_name]$sim)
          for (as in head(agent_sims, traces_max)) {
            Sys.sleep(0)
            if (isTRUE(smooth)) {
              lines(supsmu(
                dt[dt$agent == agent_name & dt$sim == as]$t,
                dt[dt$agent == agent_name & dt$sim == as][[line_data_name]]
              ),
              lwd = lwd,
              col = rgb(0.8, 0.8, 0.8, traces_alpha)
              )
            } else {
              lines(dt[dt$agent == agent_name & dt$sim == as]$t,
                    dt[dt$agent == agent_name &
                         dt$sim == as][[line_data_name]],
                    lwd = lwd,
                    col = rgb(0.8, 0.8, 0.8, traces_alpha)
              )
            }
          }
        }
      }

      if (isTRUE(use_colors)) {
        if (!is.null(disp) && !isTRUE(plot_only_disp)) {
          color <- 1
          for (agent_name in agent_levels) {
            polygon(
              c(data[data$agent == agent_name]$t, rev(data[data$agent == agent_name]$t)),
              c(data[data$agent == agent_name]$disp_lower, rev(data[data$agent == agent_name]$disp_upper)),
              col = adjustcolor(cl[color], alpha.f = 0.3),
              border = NA
            )
            color <- color + 1
          }
        }
        line_counter <- 1
        for (agent_name in agent_levels) {
          lines(
            data[data$agent == agent_name]$t,
            data[data$agent == agent_name][[line_data_name]],
            lwd  = lwd,
            lty  = lt[line_counter],
            col  = adjustcolor(cl[line_counter], alpha.f = 0.9),
            type = "l"
          )
          line_counter <- line_counter + 1
        }
      } else {
        line_counter <- 1
        for (agent_name in agent_levels) {
          if (!is.null(disp) && !isTRUE(plot_only_disp)) {
            polygon(
              c(data[data$agent == agent_name]$t, rev(data[data$agent == agent_name]$t)),
              c(data[data$agent == agent_name]$disp_lower, rev(data[data$agent == agent_name]$disp_upper)),
              col = rgb(0.8, 0.8, 0.8, 0.4),
              border = NA
            )
          }
          lines(
            data[data$agent == agent_name]$t,
            data[data$agent == agent_name][[line_data_name]],
            lwd = lwd,
            lty = lt[line_counter],
            col = rgb(0.2, 0.2, 0.2, 0.8),
            type = "l"
          )
          line_counter <- line_counter + 1
        }
      }
      axis(1)
      axis(2)
      if (is.null(xlab)) xlab = "Time step"
      title(xlab = xlab)
      if(isTRUE(plot_only_disp)) ylab_title <- paste0(ylab_title,": ",disp)
      if (is.null(ylab)) ylab = ylab_title
      title(ylab = ylab)
      box()
      if (legend) {
        if (!is.null(legend_labels)) {
          agent_levels <- legend_labels
        }
        if (!is.null(legend_border)) {
          bty <- "n"
        } else {
          bty <- "o"
        }
        if (!isTRUE(use_colors)) {
          cl <- rgb(0.2, 0.2, 0.2, 0.8)
        }
        legend(
          legend_position,
          NULL,
          agent_levels,
          col   = cl,
          title = legend_title,
          lwd   = lwd,
          lty   = lt,
          bty   = bty,
          bg    = "white"
        )
      }
      if (!isTRUE(no_par)) {
        dev.flush()
        par(old.par)
      }
    },
    gg_color_hue = function(n) {
      hues <- seq(15, 375, length = n + 1)
      hcl(h = hues, l = 65, c = 100)[1:n]
    }
  )
)

#' Plot
#'
#' Generates plots from \code{\link{History}} data.
#'
#' Usually not instantiated directly but invoked by calling the generic \code{plot(h)}, where \code{h}
#' is an \code{\link{History}} class instance.
#'
#' @name Plot
#' @aliases average optimal arms do_plot gg_color_hue check_history_data
#'
#' @section Usage:
#' \preformatted{
#'    Plot <- Plot$new()
#' }
#'
#' @section Methods:
#'
#' \describe{
#'
#'   \item{\code{cumulative(history,...)}}{
#'      Plots cumulative regret or reward (depending on parameter regret=TRUE/FALSE) over time.
#'   }
#'   \item{\code{average(history,...)}}{
#'      Plots average regret or reward (depending on parameter regret=TRUE/FALSE) over time.
#'   }
#'   \item{\code{arms(history),...}}{
#'      Plot the percentage of simulations per time step each arm was chosen over time.
#'      If multiple agents have been run, plots only the first agent.
#'   }
#'  }
#'
#' @section Plot method arguments:
#'
#' \describe{
#'   \item{\code{type}}{
#'      \code{(character, "cumulative")} Can be either "cumulative" (default), "average", or "arms".
#'      Sets the plot method when Plot() is called through R's generic plot() function.
#'      Methods are descrived in the Methods section above.
#'   }
#'   \item{\code{regret}}{
#'      \code{(logical, TRUE)} Plot policy regret (default, TRUE) or reward (FALSE)?
#'   }
#'   \item{\code{rate}}{
#'      (\code{logical, TRUE)} If rate is TRUE, the rate of regret or reward is plotted.
#'   }
#'   \item{\code{limit_agents}}{
#'      \code{(list , NULL)} Limit plotted agents to the agents in the list.
#'   }
#'   \item{\code{limit_context}}{
#'      \code{(character vector , NULL)} Only plots data where context feature name(s) in vector equal to one.
#'   }
#'   \item{\code{no_par}}{
#'      \code{(logical, FALSE)} If no_par is TRUE, Plot() does not set or adjust plotting parameters itself.
#'      This makes it possible to set custom plotting parameters through R's par() function.
#'   }
#'   \item{\code{legend}}{
#'      \code{(logical, TRUE)} Shows the legend when TRUE (default).
#'   }
#'   \item{\code{legend_title}}{
#'      \code{(character , NULL)} Sets a custom legend title.
#'   }
#'   \item{\code{legend_labels}}{
#'      \code{(character list , NULL)} Sets legend labels to custom values as specified in list.
#'   }
#'   \item{\code{legend_border}}{
#'      \code{(logical , NULL)} When TRUE, the legend is borderless.
#'   }
#'   \item{\code{legend_position}}{
#'      \code{(character , "topleft")} a single keyword from the list "bottomright", "bottom", "bottomleft",
#'      "left", "topleft", "top", "topright", "right" and "center". This places the legend on the inside of
#'      the plot frame at the given location.
#'   }
#'   \item{\code{xlim}}{
#'      \code{(c(integer,integer), NULL)} Sets x-axis limits.
#'   }
#'   \item{\code{ylim}}{
#'      \code{(c(integer,integer), NULL)} Sets y-axis limits.
#'   }
#'   \item{\code{log}}{
#'      \code{(character , "")} A character string which contains "x" if the x axis is to be logarithmic,
#'      "y" if the y axis is to be logarithmic and "xy" or "yx" if both axes are to be logarithmic.
#'   }
#'   \item{\code{use_colors}}{
#'      \code{(logical, TRUE)} If use_colors is FALSE, plots will be in grayscale.
#'      Otherwise, plots will make use of a color palette (default).
#'   }
#'   \item{\code{disp}}{
#'      \code{(character, NULL)} When disp (for "dispersion measure") is set to either 'var','sd' or 'ci',
#'      the variance, standard deviation, or 95% confidence interval will be added to the plot(s).
#'   }
#'   \item{\code{plot_only_disp}}{
#'      \code{(logical, FALSE)} When TRUE and disp is either 'var','sd' or 'ci', plot only dispersion measure.
#'   }
#'   \item{\code{traces}}{
#'      \code{(logical , FALSE)} Plot traces of independent simulations (default is FALSE).
#'   }
#'   \item{\code{traces_max}}{
#'      \code{(integer , 100)} The number of trace lines.
#'   }
#'   \item{\code{traces_alpha}}{
#'      \code{(numeric , 0.3)} Opacity of the trace lines. Default is 0.3 - that is, an opacity of 30%.
#'   }
#'   \item{\code{smooth}}{
#'      \code{(logical , FALSE)} Smooth the plot (default is FALSE)
#'   }
#'   \item{\code{interval}}{
#'      \code{(integer, NULL)} Plot only every t%%interval==0 data point.
#'   }
#'   \item{\code{cum_average}}{
#'      \code{(logical , FALSE)} Calculates moving average from cum_reward or cum_regret with step
#'      size \code{interval}.
#'   }
#'   \item{\code{color_step}}{
#'      \code{(integer, 1)} When > 1, the plot cycles through \code{nr_agents/color_step} colors.
#'   }
#'   \item{\code{lty_step}}{
#'      \code{(integer, 1)} When > 1, the plot cycles through \code{nr_agents/lty_step} line types.
#'   }
#'   \item{\code{lwd}}{
#'      \code{(integer, 1)} Line width.
#'   }
#'   \item{\code{xlab}}{
#'      \code{(character, NULL)} a title for the x axis
#'   }
#'   \item{\code{ylab}}{
#'      \code{(character, NULL)} a title for the y axis
#'   }
#'   \item{\code{trunc_over_agents}}{
#'      \code{(logical , TRUE)} Truncate the chart to the agent with the fewest time steps t.
#'   }
#'   \item{\code{trunc_per_agent}}{
#'      \code{(logical , TRUE)} Truncate every agent's plot to the number of time steps that have been fully
#'      simulated. That is, time steps for which the number of simulations equals the number defined in
#'      \code{\link{Simulator}}'s \code{simulations} parameter.
#'   }
#'  }
#'
#'
#' @seealso
#'
#' Core contextual classes: \code{\link{Bandit}}, \code{\link{Policy}}, \code{\link{Simulator}},
#' \code{\link{Agent}}, \code{\link{History}}, \code{\link{Plot}}
#'
#' Bandit subclass examples: \code{\link{BasicBernoulliBandit}}, \code{\link{ContextualLogitBandit}},
#' \code{\link{OfflineReplayEvaluatorBandit}}
#'
#' Policy subclass examples: \code{\link{EpsilonGreedyPolicy}}, \code{\link{ContextualLinTSPolicy}}
#'
#' @examples
#' \dontrun{
#'
#' bandit <- ContextualPrecachingBandit$new(weights = c(0.9, 0.1, 0.1))
#'
#' agents <- list(Agent$new(RandomPolicy$new(), bandit),
#'                Agent$new(OraclePolicy$new(), bandit),
#'                Agent$new(ThompsonSamplingPolicy$new(1.0, 1.0), bandit),
#'                Agent$new(Exp3Policy$new(0.1), bandit),
#'                Agent$new(GittinsBrezziLaiPolicy$new(), bandit),
#'                Agent$new(UCB1Policy$new(), bandit))
#'
#' history <- Simulator$new(agents, horizon = 100, simulations = 1000)$run()
#'
#' par(mfrow = c(3, 2), mar = c(1, 4, 2, 1), cex=1.3)
#'
#' plot(history, type = "cumulative", use_colors = FALSE, no_par = TRUE, legend_border = FALSE,
#'      limit_agents = c("GittinsBrezziLai", "UCB1","ThompsonSampling"))
#'
#' plot(history, type = "cumulative", regret = FALSE, legend = FALSE,
#'      limit_agents = c("UCB1"), traces = TRUE, no_par = TRUE)
#'
#' plot(history, type = "cumulative", regret = FALSE, rate = TRUE, disp = "sd",
#'      limit_agents = c("Exp3", "ThompsonSampling"),
#'      legend_position = "bottomright", no_par = TRUE)
#'
#' plot(history, type = "cumulative", rate = TRUE, plot_only_disp = TRUE,
#'      disp = "var", smooth = TRUE, limit_agents = c("UCB1", "GittinsBrezziLai"),
#'     legend_position = "bottomleft", no_par = TRUE)
#'
#' plot(history, type = "average", disp = "ci", regret = FALSE, interval = 10,
#'      smooth = TRUE, legend_position = "bottomright", no_par = TRUE, legend = FALSE)
#'
#' plot(history, limit_agents = c("ThompsonSampling"), type = "arms",
#'      interval = 20, no_par = TRUE)
#'
#' }
#'
#'
NULL
