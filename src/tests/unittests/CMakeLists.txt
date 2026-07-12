add_subdirectory(iop)

if(USE_AI)
  add_subdirectory(ai)
endif(USE_AI)

if(USE_VULKAN)
  add_subdirectory(common)
endif(USE_VULKAN)

add_cmocka_test(test_sample
                SOURCES test_sample.c
                LINK_LIBRARIES cmocka)

# Windows: main-method requires the wrapper provided by lib-darktable
if(WIN32)
    target_link_libraries(test_sample PRIVATE lib_darktable)
    _copy_required_library(test_sample lib_darktable)
endif(WIN32)
