# ---------- master mapping: per-dataset mappings for educ_c and educlev ----------
# categories:
# 1 -> "gimnazjalne, podstawowe I niższe"
# 2 -> "zasadnicze zawodowe"
# 3 -> "średnie"
# 4 -> "wyższe"

edu_cat_labels <- c(
  "1" = "gimnazjalne, podstawowe I niższe",
  "2" = "zasadnicze zawodowe",
  "3" = "średnie",
  "4" = "wyższe"
)

# Build a list where each element is a list(educ_c = named_int_vector, educlev = named_int_vector)
# Keys in the named vectors are the original codes as strings.
mapping_per_dataset <- list(

  # 1986 (pl86p) -- older proprietary codes 0..8
  pl86p = list(
    educ_c = c(
      "0" = NA_integer_,   # missing
      "1" = 4L,            # university w/ degree -> wyższe
      "2" = 4L,            # university w/o degree -> wyższe (incomplete tertiary)
      "3" = 3L,            # general level complete -> średnie
      "4" = 1L,            # general level incomplete -> below upper-secondary (treat as primary/lower)
      "5" = 2L,            # occupational training complete -> vocational
      "6" = 3L,            # grammar school complete -> średnie
      "7" = 1L,            # grammar school incomplete -> lower/primary
      "8" = NA_integer_    # other/ambiguous -> NA
    ),
    educlev = c(
      "110" = 1L, "120" = 1L, "210" = 3L, "300" = 4L
    )
  ),

  # 1992 (pl92p) and 1995 (pl95p) use 1..15 scheme (students etc.)
  pl92p = list(
    educ_c = c(
      "1" = 4L,  # complete higher
      "2" = 4L,  # incomplete higher
      "3" = 3L,  # post-secondary -> treat as średnie / post-secondary non-tertiary
      "4" = 3L,  # complete secondary -> średnie
      "5" = 1L,  # incomplete secondary -> below upper-secondary
      "6" = 2L,  # basic vocational -> zasadnicze zawodowe
      "7" = 1L,  # complete elementary -> podstawowe
      "8" = 1L,  # incomplete elementary -> podstawowe i niższe
      "9" = 1L,  # self-taught/illiterate/pre-school -> place in low group
      "10" = 1L, # elementary school students -> low
      "11" = 2L, # basic vocational school students -> zasadnicze zawodowe
      "12" = 3L, # secondary school students -> średnie
      "13" = 3L, # secondary vocational students -> średnie
      "14" = 3L, # post-secondary school students -> średnie/post-secondary
      "15" = 4L  # university students -> wyższe
    ),
    educlev = c(
      "110" = 1L, "111" = 1L, "120" = 1L,
      "210" = 3L, "220" = 3L, "300" = 4L
    )
  ),
  pl95p = mapping_per_dataset[["pl92p"]], # same scheme for 1995

  # 1999 (pl99p) — ISCED-like but missing some lower-secondary codes
  pl99p = list(
    educ_c = c(
      "1"  = 1L,   # here 1 = no formal education (as in your dump) -> low
      "10" = 1L,   # primary
      "31" = 2L,   # basic vocational
      "32" = 3L,   # upper secondary general
      "33" = 3L,   # upper secondary vocational
      "40" = 3L,   # post-secondary non-tertiary -> średnie group
      "50" = 4L    # tertiary coarse -> wyższe
    ),
    educlev = c(
      "110" = 1L, "120" = 1L, "210" = 3L, "220" = 3L, "300" = 4L
      # note: 130 (lower-secondary) absent in 1999 — no synthetic mapping here
    )
  ),

  # 2004-2009 (pl04p..pl09p) ISCED-like with coarse tertiary (50)
  pl04p = list(
    educ_c = c("1"=1L, "10"=1L, "20"=1L, "31"=2L, "32"=3L, "33"=3L, "40"=3L, "50"=4L),
    educlev = c("110"=1L,"120"=1L,"130"=1L,"210"=3L,"220"=3L,"300"=4L)
  ),
  pl05p = mapping_per_dataset[["pl04p"]],
  pl06p = mapping_per_dataset[["pl04p"]],
  pl07p = mapping_per_dataset[["pl04p"]],
  pl08p = mapping_per_dataset[["pl04p"]],
  pl09p = mapping_per_dataset[["pl04p"]],

  # 2010 onwards (pl10p..pl23p) — ISCED-like with split tertiary (51,52,53,60)
  pl10p = list(
    educ_c = c("1"=1L,"10"=1L,"20"=1L,"31"=2L,"32"=3L,"33"=3L,"40"=3L,
               "51"=4L,"52"=4L,"53"=4L,"60"=4L),
    educlev = c(
      "110"=1L,"111"=1L,"120"=1L,"130"=1L,
      "210"=3L,"220"=3L,"200"=3L,
      "311"=4L,"312"=4L,"313"=4L,"320"=4L
    )
  )
)

# reuse the same mapping for 2011..2018 and 2019..2023 (they follow same scheme)
for (yr in c("pl11p","pl12p","pl13p","pl14p","pl15p","pl16p","pl17p","pl18p",
             "pl19p","pl20p","pl21p","pl22p","pl23p")) {
  mapping_per_dataset[[yr]] <- mapping_per_dataset[["pl10p"]]
}

# 1992/1995 mapping already stored above; ensure names exist
mapping_per_dataset[["pl92p"]] <- mapping_per_dataset[["pl92p"]]
mapping_per_dataset[["pl95p"]] <- mapping_per_dataset[["pl95p"]]

# ---------- helper function to apply mapping to a data.frame ----------
# df: data.frame with columns educ_c and/or educlev
# ds_tag: one of names(mapping_per_dataset), e.g. "pl10p"
# returns df with edu_cat_num (1..4 integer) and edu_cat_label (Polish label)
harmonize_edu <- function(df, ds_tag) {
  if (!ds_tag %in% names(mapping_per_dataset)) {
    stop("Unknown dataset tag. Available tags: ", paste(names(mapping_per_dataset), collapse = ", "))
  }
  maps <- mapping_per_dataset[[ds_tag]]

  # start with NA
  df$edu_cat_num <- as.integer(NA)

  # try educ_c first (preferred)
  if ("educ_c" %in% names(df) && !is.null(maps$educ_c)) {
    ec_keys <- as.character(df$educ_c)
    # safe lookup: will yield NA where key not present
    mapped_from_c <- unname(maps$educ_c[ec_keys])
    # convert to integer, keep NA where missing
    df$edu_cat_num <- as.integer(mapped_from_c)
  }

  # fallback: for rows still NA, try educlev mapping
  missing_idx <- which(is.na(df$edu_cat_num))
  if (length(missing_idx) > 0 && "educlev" %in% names(df) && !is.null(maps$educlev)) {
    el_keys <- as.character(df$educlev[missing_idx])
    mapped_from_el <- unname(maps$educlev[el_keys])
    df$edu_cat_num[missing_idx] <- as.integer(mapped_from_el)
  }

  # keep as NA if still unmapped
  df$edu_cat_label <- ifelse(is.na(df$edu_cat_num),
                             NA_character_,
                             edu_cat_labels[as.character(df$edu_cat_num)])
  df
}

# ---------- Example usage ----------
# df <- read.csv("pl10p_data.csv")   # or however you load a dataset
# df <- harmonize_edu(df, "pl10p")
# table(df$edu_cat_num, useNA = "ifany")