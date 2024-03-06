# Main App

## Summary

The __init__ method initializes an instance of the App class. It sets up the necessary directories, logger, and other configurations based on the provided prefix and command line arguments.
### Example Usage
`
app = App(prefix="myapp", args=AppArgs().default())
`
### Code Analysis
##### Inputs
`
prefix (str): The prefix for the application.
args (AppArgs): Command line arguments for the application.
`

### Flow
- Get the absolute path of the current file and determine the system flag (Linux, Mac, or Windows).
- Set the current working directory to the directory of the current file.
- Set the start directory to the current working directory.
- Set the path for the last used application prefix file.
- If no prefix is provided, check if the last used prefix exists in the file and use it as the prefix.
- If a prefix is provided, write it to the last used prefix file.
- Set the application ID by combining the prefix and the node name.
- Set up the necessary directories for data, configuration, and information files.
- Print the start directory and set up the logger.
- If the init argument is provided, add the start directory to the system path and initialize the toolbox.
- Set up the keys and defaults for the configuration file.
- Load the configuration file and get the debug mode value.
- Set up the runnable, development mode, and functions attributes.
- Set up the interface type, prefix, module list, and other attributes.
- Print the system information and version.
- If the update argument is provided, pull the latest changes from the git repository.
- If the get_version argument is provided, get the version of the specified module.
- Log the completion of the initialization.

### Outputs
An instance of the App class with the necessary configurations and attributes set up.
