## 1 Generating a new SSH key pair
To generate a new SSH key, use the following command:
* GNU/Linux / macOS:   
  `ssh-keygen -t rsa -C "GitLab" -b 4096`
  
## 2 Locating an existing SSH key pair
GNU/Linux / macOS / PowerShell:   
`cat ~/.ssh/id_rsa.pub`
