#pragma once

#include <fcntl.h>
#include <unistd.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

std::string getTimestamp() {
  // Get the current time point
  auto now = std::chrono::system_clock::now();

  // Convert it to milliseconds since the epoch
  long long timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                            now.time_since_epoch())
                            .count();

  // Convert the timestamp to string
  return std::to_string(timestamp);
}

class CStdoutRedirector {
 public:
  enum MODE { FILE, PIPE };
  enum PIPES { READ, WRITE };

  CStdoutRedirector(const MODE mode, const std::string& id)
      : mode(mode), filename("./" + id + ".tmp." + getTimestamp()) {}
  ~CStdoutRedirector();

  void init();
  void start();
  void stop();
  std::string get() const { return redirected_output; }
  void clear() { redirected_output.clear(); };

 private:
  const MODE mode;

  int fd;
  int cur_pipe[2];
  int old_stdout = 0;
  int old_stderr = 0;
  bool redirecting = false;
  std::string redirected_output;

  const std::string filename;
};

void CStdoutRedirector::init() {
  if (mode == FILE) {
    fd = open(filename.c_str(), O_WRONLY | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR);
    if (fd <= 0) {
      std::cout << "WARNING: can't open file " << filename << std::endl;
      std::exit(-1);
    }
    old_stdout = dup(fileno(stdout));

  } else if (mode == PIPE) {
    cur_pipe[READ] = 0;
    cur_pipe[WRITE] = 0;
    if (pipe(cur_pipe) != -1) {
      old_stdout = dup(fileno(stdout));
      old_stderr = dup(fileno(stderr));
    }
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);

  } else {
    std::cout << "ERROR: not recognised mode" << std::endl;
    std::exit(-1);
  }
}

CStdoutRedirector::~CStdoutRedirector() {
  if (mode == FILE) {
    if (redirecting) stop();
    if (old_stdout > 0) close(old_stdout);
    if (fd > 0) close(fd);
    remove(filename.c_str());

  } else if (mode == PIPE) {
    if (redirecting) stop();
    if (old_stdout > 0) close(old_stdout);
    if (old_stderr > 0) close(old_stderr);
    if (cur_pipe[READ] > 0) close(cur_pipe[READ]);
    if (cur_pipe[WRITE] > 0) close(cur_pipe[WRITE]);
  }
}

void CStdoutRedirector::start() {
  if (mode == FILE) {
    if (fd <= 0 || redirecting) return;
    dup2(fd, fileno(stdout));
    redirecting = true;

  } else if (mode == PIPE) {
    if (redirecting) return;
    dup2(cur_pipe[WRITE], fileno(stdout));
    dup2(cur_pipe[WRITE], fileno(stderr));
    redirecting = true;
  }
  return;
}

void CStdoutRedirector::stop() {
  // stop
  if (!redirecting) return;

  redirecting = false;
  dup2(old_stdout, fileno(stdout));

  if (mode == PIPE) dup2(old_stderr, fileno(stderr));

  // get the output
  if (mode == FILE) {
    if (fd) close(fd);

    std::ifstream in(filename, std::ios::in | std::ios::binary);
    std::ostringstream contents;
    contents << in.rdbuf();
    in.close();
    redirected_output += contents.str();

    remove(filename.c_str());
    fd = open(filename.c_str(), O_WRONLY | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR);
    if (fd <= 0) {
      std::cout << "WARNING: can't open file" << std::endl;
      std::exit(-1);
    }

  } else if (mode == PIPE) {
    constexpr size_t buf_size = 1024 * 16;
    char buf[buf_size];
    fcntl(cur_pipe[READ], F_SETFL, O_NONBLOCK);
    size_t bytes_read = read(cur_pipe[READ], buf, buf_size - 1);
    redirected_output.reserve(bytes_read + redirected_output.size());
    redirected_output += buf;
  }

  return;
}