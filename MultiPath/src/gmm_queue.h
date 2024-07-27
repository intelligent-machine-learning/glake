#pragma once
#include <condition_variable>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>

// fifo_queue: thread safe, concurrent push at tail and pop from front
// useful for object caching
// ref: https://blog.csdn.net/u013362955/article/details/120313008
//      https://blog.csdn.net/mymodian9612/article/details/53608084
// example:
// fifo_queue<int> idx_cache[256];
// idx_cache[256].push(j);
// int *val = idx_cache[i].pop().get();
//
// fifo_queue<A *> obj_cache;
// obj_cache.push(new A(i)); //populate the cache
// std::shared_ptr<A *> obj= obj_cache.pop(); //get obj
// cout << ' ' << (*(obj.get()))->getValue();
// obj_cache.push(*(obj.get())); //return obj

template <typename T>
class fifo_queue {
 private:
  struct node {
    std::shared_ptr<T> data;
    std::unique_ptr<node> next;
  };

  std::mutex head_mut;
  std::mutex tail_mut;

  std::unique_ptr<node> head;
  node* tail;
  std::condition_variable data_cond;
  uint32_t cnt;

  node* get_tail() {
    std::lock_guard<std::mutex> lock(tail_mut);
    return tail;
  }

 public:
  fifo_queue() : head(new node), tail(head.get()), cnt(0) {}

  fifo_queue(const fifo_queue&) = delete;
  fifo_queue operator=(const fifo_queue&) = delete;
  ~fifo_queue() = default;

  // push into tail
  void push(T t) {
    std::shared_ptr<T> new_data(std::make_shared<T>(std::move(t)));
    std::unique_ptr<node> new_tail(new node);
    node* const p = new_tail.get();
    {
      std::lock_guard<std::mutex> lock(tail_mut);
      tail->data = new_data;
      tail->next = std::move(new_tail);
      tail = p;
      cnt++;
    }
    data_cond.notify_one();
    // std::cout << "push new, cnt:" << cnt<<std::endl;
  }

  /*
  void push(T &&t) {
    std::shared_ptr<T> new_data(std::make_shared<T>(std::forward<T>(t)));
    std::unique_ptr<node> new_tail(new node);
    node * const p = new_tail.get();
    {
      std::lock_guard<std::mutex> lock(tail_mut);
      tail->data = new_data;
      tail->next = std::move(new_tail);
      tail = p;
      cnt++;
    }
    data_cond.notify_one();
    //std::cout << "push new, cnt:" << cnt<<std::endl;
  }
 */

  // pop from front, blocking
  std::shared_ptr<T> pop() {
    std::lock_guard<std::mutex> h_lock(head_mut);

    {
      std::unique_lock<std::mutex> t_lock(tail_mut);
      data_cond.wait(t_lock, [&] { return head.get() != tail; });
    }

    auto old_head = std::move(head);
    head = std::move(old_head->next);
    cnt--;

    return old_head->data;
  }

  // pop from front, no wait
  std::shared_ptr<T> try_pop() {
    std::lock_guard<std::mutex> h_lock(head_mut);

    if (head.get() == get_tail()) return nullptr;

    auto old_head = std::move(head);
    head = std::move(old_head->next);
    cnt--;

    return old_head->data;
  }

  // get data from head, not remove
  // the caller shall take another lock or ensure no race
  std::shared_ptr<T> try_read() {
    std::lock_guard<std::mutex> h_lock(head_mut);

    if (head.get() == get_tail()) return nullptr;

    return head->data;
  }

  // check node state via a given func, ensure return status is met before pop
  /*
  typedef cudaError_t (*valid_func) (T node);
  std::shared_ptr<T> try_pop_if(valid_func f, cudaError_t cond) {
    std::lock_guard<std::mutex> h_lock(head_mut);

    if (head.get() == get_tail())
      return nullptr;

    if (cond == f(*head->data.get())) {
      auto old_head = std::move(head);
      head = std::move(old_head->next);
      cnt--;

      return old_head->data;
    } else {
      return nullptr;
    }

  }
  */

  // empty or not
  bool empty() {
    std::lock_guard<std::mutex> h_lock(head_mut);
    return (head.get() == get_tail());
  }
};

/*
template<typename T>
class fifo_queue<T *> {
 private:
   struct node {
     T *data;
     node *next;
   };

   std::mutex head_mut;
   std::mutex tail_mut;

   node *head;
   node *tail;
   std::condition_variable data_cond;
   uint32_t cnt;

   node* get_tail() {
     std::lock_guard<std::mutex> lock(tail_mut);;
     return tail;
   }

  typedef cudaError_t (*valid_func) (T node);

 public:
   fifo_queue(): head(new node), tail(head), cnt(0) {}

   fifo_queue(const fifo_queue&) = delete;
   fifo_queue operator=(const fifo_queue&)  = delete;
   ~fifo_queue() = default;

   // push into tail
   void push(T *t) {
     node *new_tail  = new node;
     {
       std::lock_guard<std::mutex> lock(tail_mut);
       tail->data = t;
       tail->next = new_tail;
       tail = new_tail;
       cnt++;
     }
     data_cond.notify_one();
     //std::cout << "push new, cnt:" << cnt<<std::endl;
   }

   // pop from front, blocking
   T* pop() {
     node *old_head = nullptr;

     {
       std::unique_lock<std::mutex> h_lock(head_mut);
       {
         std::unique_lock<std::mutex> t_lock(tail_mut);
         data_cond.wait(t_lock,[&]()
         {
           return head.get() != tail;
         });
       }
       old_head = head;
       head = head->next;
       cnt--;
     }

     T *data = old_head->data;
     delete old_head;

     return data;
   }

   // pop from front, no wait
   T* try_pop() {
     node *old_head = nullptr;
     {
       std::unique_lock<std::mutex> h_lock(head_mut);
       if (head == get_tail())
         return nullptr;

       old_head = head;
       head = head->next;
       cnt--;
     }

     T *data = old_head->data;
     delete old_head;

     return data;
   }

   // check node state via a given func, ensure return status is met before pop
   T* try_pop_if(valid_func f, cudaError_t expect) {
     node *old_head = nullptr;

     {
       std::unique_lock<std::mutex> h_lock(head_mut);
       if (head == get_tail())
         return nullptr;

       T *data_ = head->data;
       if (expect == f(data_)) {
         old_head = head;
         head = head->next;
         cnt--;
       }
     }
     T *data = old_head->data;
     delete old_head;
     return data;
   }

   // empty or not
   bool empty() {
     std::lock_guard<std::mutex> h_lock(head_mut);
     return (head == get_tail());
   }
};
*/
