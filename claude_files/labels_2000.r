# Mapping educ_c and educlev (pl86p..pl23p) -> 1..5
# Categories:
# 1 = "gimnazjalne, podstawowe I niższe"
# 2 = "zasadnicze zawodowe/branżowe"
# 3 = "średnie ogólnokształcące"
# 4 = "policealne oraz średnie zawodowe/branżowe"
# 5 = "wyższe"

edu_cat_labels <- c(
  "1" = "gimnazjalne, podstawowe I niższe",
  "2" = "zasadnicze zawodowe/branżowe",
  "3" = "średnie ogólnokształcące",
  "4" = "policealne oraz średnie zawodowe/branżowe",
  "5" = "wyższe"
)

# Per-dataset mapping (keys are strings)
mapping_per_dataset <- list(

  # 1986: codes 0..8 (older proprietary scheme)
  pl86p = list(
    educ_c = c(
      "0" = NA_integer_,   # missing/other
      "1" = 5L,            # university w/ degree -> wyższe
      "2" = 5L,            # university w/o degree -> wyższe
      "3" = 3L,            # general level complete -> średnie ogólnokształcące
      "4" = 1L,            # general level incomplete -> gimnazjalne/podstawowe i niższe
      "5" = 2L,            # occ training complete -> zasadnicze zawodowe
      "6" = 3L,            # grammar school complete -> średnie ogólnokształcące
      "7" = 1L,            # grammar school incomplete -> gimnazjalne/podstawowe i niższe
      "8" = NA_integer_    # other/ambiguous -> NA
    ),
    educlev = c(
      "110" = 1L, "120" = 1L, "210" = 3L, "300" = 5L
    )
  ),

  # 1992 and 1995: codes 1..15 (fine-grained)
  pl92p = list(
    educ_c = c(
      "1"  = 5L,  # complete higher -> wyższe
      "2"  = 5L,  # incomplete higher -> wyższe
      "3"  = 4L,  # post-secondary -> policealne / post-sec non-tertiary
      "4"  = 3L,  # complete secondary -> średnie ogólnokształcące (default)
      "5"  = 1L,  # incomplete secondary -> gimnazjalne/podstawowe i niższe
      "6"  = 2L,  # basic vocational -> zasadnicze zawodowe/branżowe
      "7"  = 1L,  # complete elementary -> podstawowe i niższe
      "8"  = 1L,  # incomplete elementary -> podstawowe i niższe
      "9"  = 1L,  # self-taught / illiterate / pre-school -> podstawowe i niższe
      "10" = 1L,  # elementary school students -> podstawowe i niższe
      "11" = 2L,  # basic vocational school students -> zasadnicze zawodowe
      "12" = 3L,  # secondary school students -> średnie ogólnokształcące
      "13" = 4L,  # secondary vocational school students -> policealne/średnie zawodowe
      "14" = 4L,  # post-secondary school students -> policealne
      "15" = 5L   # university students -> wyższe
    ),
    educlev = c(
      "110" = 1L, "111" = 1L, "120" = 1L,
      "210" = 3L, "220" = 4L, "300" = 5L
    )
  ),
  pl95p = mapping_per_dataset[["pl92p"]], # same scheme for 1995

  # 1999: ISCED-like but missing some lower-secondary codes
  pl99p = list(
    educ_c = c(
      "1"  = 1L,  # no formal education -> podstawowe i niższe
      "10" = 1L,  # primary -> podstawowe i niższe
      "31" = 2L,  # basic vocational -> zasadnicze zawodowe
      "32" = 3L,  # upper secondary general -> średnie ogólnokształcące
      "33" = 4L,  # upper secondary vocational -> policealne/średnie zawodowe
      "40" = 4L,  # post-secondary non-tertiary -> policealne
      "50" = 5L   # tertiary coarse -> wyższe
    ),
    educlev = c(
      "110" = 1L, "120" = 1L, "210" = 3L, "220" = 4L, "300" = 5L
      # 130 absent in this year
    )
  ),

  # 2004-2009: ISCED-like with coarse tertiary (50)
  pl04p = list(
    educ_c = c("1"=1L, "10"=1L, "20"=1L, "31"=2L, "32"=3L, "33"=4L, "40"=4L, "50"=5L),
    educlev = c("110"=1L,"120"=1L,"130"=1L,"210"=3L,"220"=4L,"300"=5L)
  ),
  pl05p = mapping_per_dataset[["pl04p"]],
  pl06p = mapping_per_dataset[["pl04p"]],
  pl07p = mapping_per_dataset[["pl04p"]],
  pl08p = mapping_per_dataset[["pl04p"]],
  pl09p = mapping_per_dataset[["pl04p"]],

  # 2010 onwards: ISCED-like with split tertiary (51,52,53,60)
  pl10p = list(
    educ_c = c("1"=1L,"10"=1L,"20"=1L,"31"=2L,"32"=3L,"33"=4L,"40"=4L,
               "51"=5L,"52"=5L,"53"=5L,"60"=5L),
    educlev = c(
      "110"=1L,"111"=1L,"120"=1L,"130"=1L,
      "210"=3L,"220"=4L,"200"=4L,
      "311"=5L,"312"=5L,"313"=5L,"320"=5L
    )
  )
)

# replicate pl10p mapping for pl11p..pl23p (they use same modern scheme)
for (yr in c("pl11p","pl12p","pl13p","pl14p","pl15p","pl16p","pl17p","pl18p",
             "pl19p","pl20p","pl21p","pl22p","pl23p")) {
  mapping_per_dataset[[yr]] <- mapping_per_dataset[["pl10p"]]
}

# ensure pl92p and pl95p entries exist (already set above)
mapping_per_dataset[["pl92p"]] <- mapping_per_dataset[["pl92p"]]
mapping_per_dataset[["pl95p"]] <- mapping_per_dataset[["pl95p"]]

# ---------- helper: apply mapping ----------
# df: data.frame with columns educ_c and/or educlev
# ds_tag: dataset tag, e.g. "pl10p"
# returns df with edu_cat_num (1..5) and edu_cat_label (Polish)
harmonize_edu <- function(df, ds_tag) {
  if (!ds_tag %in% names(mapping_per_dataset)) {
    stop("Unknown dataset tag. Available tags: ", paste(names(mapping_per_dataset), collapse = ", "))
  }
  maps <- mapping_per_dataset[[ds_tag]]

  df$edu_cat_num <- as.integer(NA)

  # prefer educ_c mapping if present
  if ("educ_c" %in% names(df) && !is.null(maps$educ_c)) {
    ec_keys <- as.character(df$educ_c)
    mapped_from_c <- unname(maps$educ_c[ec_keys])
    df$edu_cat_num <- as.integer(mapped_from_c)
  }

  # fallback to educlev for still-NA rows
  missing_idx <- which(is.na(df$edu_cat_num))
  if (length(missing_idx) > 0 && "educlev" %in% names(df) && !is.null(maps$educlev)) {
    el_keys <- as.character(df$educlev[missing_idx])
    mapped_from_el <- unname(maps$educlev[el_keys])
    df$edu_cat_num[missing_idx] <- as.integer(mapped_from_el)
  }

  df$edu_cat_label <- ifelse(is.na(df$edu_cat_num),
                             NA_character_,
                             edu_cat_labels[as.character(df$edu_cat_num)])
  df
}

# Example:
# df <- read.csv("pl10p_data.csv")
# df <- harmonize_edu(df, "pl10p")
# table(df$edu_cat_num, useNA = "ifany")