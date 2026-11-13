find_package(Doxygen 1.9.5 REQUIRED)

include(FetchContent)
FetchContent_Declare(
        doxygen-awesome-css
        GIT_REPOSITORY https://github.com/jothepro/doxygen-awesome-css.git
        GIT_TAG v2.5.0
)
FetchContent_MakeAvailable(doxygen-awesome-css)
FetchContent_GetProperties(doxygen-awesome-css SOURCE_DIR NUM_DOXYGEN_AWESOME_CSS_DIR)

get_property(NUM_INCLUDE_DIRS GLOBAL PROPERTY NUM_INCLUDE_DIRS)

set(DOXYGEN_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/docs")
set(DOXYGEN_USE_MDFILE_AS_MAINPAGE "${PROJECT_SOURCE_DIR}/docs/mainpage.md")
set(DOXYGEN_STRIP_FROM_INC_PATH ${NUM_INCLUDE_DIRS})
set(DOXYGEN_EXTRACT_ALL YES)
set(DOXYGEN_EXCLUDE_SYMBOLS "*::detail")
set(DOXYGEN_BRIEF_MEMBER_DESC NO)
set(DOXYGEN_SORT_MEMBER_DOCS NO)
set(DOXYGEN_BUILTIN_STL_SUPPORT YES)
set(DOXYGEN_DOT_IMAGE_FORMAT svg)
set(DOXYGEN_HIDE_UNDOC_RELATIONS NO)
set(DOXYGEN_USE_MATHJAX YES)
set(DOXYGEN_MATHJAX_VERSION MathJax_3)
set(DOXYGEN_WARN_AS_ERROR FAIL_ON_WARNINGS)

set(DOXYGEN_HTML_EXTRA_STYLESHEET
        "${NUM_DOXYGEN_AWESOME_CSS_DIR}/doxygen-awesome.css"
        "${NUM_DOXYGEN_AWESOME_CSS_DIR}/doxygen-awesome-sidebar-only.css"
)
set(DOXYGEN_GENERATE_TREEVIEW YES)
set(DOXYGEN_DISABLE_INDEX NO)
set(DOXYGEN_FULL_SIDEBAR NO)
set(DOXYGEN_HTML_COLORSTYLE LIGHT)

doxygen_add_docs(docs
        ${NUM_INCLUDE_DIRS}
        "${PROJECT_SOURCE_DIR}/docs/groups.dox"
        "${PROJECT_SOURCE_DIR}/docs/mainpage.md"
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        COMMENT "Generating API documentation"
)
