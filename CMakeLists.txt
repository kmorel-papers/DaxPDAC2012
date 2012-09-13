cmake_minimum_required(VERSION 2.8)

project(DaxPDAC2012 NONE)

include(UseLATEX.cmake)

add_latex_document(DaxPDAC2012.tex
  BIBFILES DaxPDAC2012.bib
  )
