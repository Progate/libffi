/* -----------------------------------------------------------------------
   ffi.c - Copyright (c) 2018-2023  Hood Chatham, Brion Vibber, Kleis Auke Wolthuizen, and others.
           Copyright (c) 2026  Anthony Green
           Copyright (c) 2026  Progate, Inc.

   wasm32 Foreign Function Interface for BrowserOS (wasm32-wasip1 and
   wasm32-wasip1-threads with browser-dyld).

   Permission is hereby granted, free of charge, to any person obtaining
   a copy of this software and associated documentation files (the
   ``Software''), to deal in the Software without restriction, including
   without limitation the rights to use, copy, modify, merge, publish,
   distribute, sublicense, and/or sell copies of the Software, and to
   permit persons to whom the Software is furnished to do so, subject to
   the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED ``AS IS'', WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
   NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
   HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
   WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
   ----------------------------------------------------------------------- */

/*
 * This replaces libffi's src/wasm/ffi.c, which does the same lowering in
 * Emscripten's JavaScript (EM_JS). Here the C ABI lowering stays in C and only
 * the two things wasm code cannot do itself go to the host (browser-dyld,
 * namespace "browser_dyld"):
 *
 *   - calling a function pointer whose type is only known at run time
 *     (call_indirect needs a static type): ffi_call
 *   - making a new function pointer of a given type (a closure): the host
 *     builds a wasm function of that type that calls back into
 *     __browseros_ffi_closure_call, and puts it in the indirect function table
 *     of every thread
 *
 * The lowering is clang's wasm32 C ABI, the same as Emscripten's:
 *   - integers up to 32 bits and pointers are i32, 64-bit integers i64, float
 *     f32, double f64, long double two i64 (low, high)
 *   - a struct with a single member (recursively, up to 16 bytes) is passed and
 *     returned as that member; any other struct argument is passed as a
 *     pointer to a copy, and any other struct or long double result is
 *     returned through a pointer passed as a hidden first argument
 *   - variadic arguments are laid out in a buffer in order, each at its
 *     natural alignment (structs as pointers to copies), and a pointer to the
 *     buffer is passed as one extra argument
 *
 * A lowered call is described by a signature string, one letter per value
 * ('v' void result, 'i' i32, 'j' i64, 'f' f32, 'd' f64; the first letter is
 * the result), and an array of 8-byte slots holding the arguments.
 */

#include <ffi.h>
#include <ffi_common.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define VARARGS_FLAG 1
// WebAssembly's JS API limits a function type to 1000 parameters, and calls and
// closures are made through function types of the lowered signature.
#define MAX_ARGS 1000

__attribute__((import_module("browser_dyld"), import_name("ffi_call")))
int32_t __browseros_ffi_call(uint32_t fn, const char *signature, uint32_t length,
                             const uint64_t *slots, void *result);
__attribute__((import_module("browser_dyld"), import_name("ffi_closure_reserve")))
int32_t __browseros_ffi_closure_reserve(void);
__attribute__((import_module("browser_dyld"), import_name("ffi_closure_set")))
int32_t __browseros_ffi_closure_set(uint32_t index, const char *signature, uint32_t length,
                                    void *closure);
__attribute__((import_module("browser_dyld"), import_name("ffi_closure_release")))
void __browseros_ffi_closure_release(uint32_t index);

static ffi_type *unbox(ffi_type *type);
static int returns_by_pointer(ffi_type *rtype);

ffi_status FFI_HIDDEN
ffi_prep_cif_machdep(ffi_cif *cif)
{
  if (cif->abi != FFI_WASM32)
    return FFI_BAD_ABI;
  // A variadic cif comes here from ffi_prep_cif_machdep_var, which has set nfixedargs.
  if (!(cif->flags & VARARGS_FLAG))
    cif->nfixedargs = cif->nargs;
  if (cif->rtype->type == FFI_TYPE_COMPLEX)
    return FFI_BAD_TYPEDEF;
  for (unsigned i = 0; i < cif->nargs; i++)
    if (cif->arg_types[i]->type == FFI_TYPE_COMPLEX)
      return FFI_BAD_TYPEDEF;
  // The lowered function type must fit the WebAssembly limit on parameters:
  // the hidden result pointer, two values per long double, and the pointer to
  // the variadic arguments all count.
  size_t lowered = returns_by_pointer(unbox(cif->rtype)) ? 1 : 0;
  for (unsigned i = 0; i < cif->nfixedargs; i++) {
    unsigned short type = unbox(cif->arg_types[i])->type;
    lowered += type == FFI_TYPE_LONGDOUBLE ? 2 : type == FFI_TYPE_VOID ? 0 : 1;
  }
  if (cif->flags & VARARGS_FLAG)
    lowered += 1;
  if (lowered > MAX_ARGS)
    return FFI_BAD_TYPEDEF;
  return FFI_OK;
}

ffi_status FFI_HIDDEN
ffi_prep_cif_machdep_var(ffi_cif *cif, unsigned nfixedargs, unsigned ntotalargs)
{
  (void)ntotalargs;
  cif->flags |= VARARGS_FLAG;
  cif->nfixedargs = nfixedargs;
  // libffi calls only this one for a variadic cif (FFI_TARGET_SPECIFIC_VARIADIC),
  // so run the same checks as for a fixed one.
  return ffi_prep_cif_machdep(cif);
}

/*
 * A struct with exactly one member is passed as that member (recursively), as
 * long as it is at most 16 bytes. Larger ones stay structs: CPython describes
 * some of them with the size of a pointer (see _ctypes/stgdict.c), and the
 * true ABI for them is indeed a pointer. An empty struct is void.
 */
static ffi_type *
unbox(ffi_type *type)
{
  while (type->type == FFI_TYPE_STRUCT && type->size <= 16) {
    ffi_type **elements = type->elements;
    if (elements[0] == NULL)
      return &ffi_type_void;
    if (elements[1] != NULL)
      break;
    type = elements[0];
  }
  return type;
}

/* The signature letter of a lowered scalar. */
static char
letter_of(unsigned short type)
{
  switch (type) {
  case FFI_TYPE_UINT64:
  case FFI_TYPE_SINT64:
    return 'j';
  case FFI_TYPE_FLOAT:
    return 'f';
  case FFI_TYPE_DOUBLE:
    return 'd';
  default:
    return 'i';
  }
}

static int
returns_by_pointer(ffi_type *rtype)
{
  return rtype->type == FFI_TYPE_STRUCT || rtype->type == FFI_TYPE_LONGDOUBLE;
}

/* Reads one scalar argument into a slot, extending narrow integers as clang does. */
static uint64_t
load_scalar(unsigned short type, const void *value)
{
  uint64_t slot = 0;
  switch (type) {
  case FFI_TYPE_UINT8:
    return *(const uint8_t *)value;
  case FFI_TYPE_SINT8:
    return (uint32_t)(int32_t) * (const int8_t *)value;
  case FFI_TYPE_UINT16:
    return *(const uint16_t *)value;
  case FFI_TYPE_SINT16:
    return (uint32_t)(int32_t) * (const int16_t *)value;
  case FFI_TYPE_UINT64:
  case FFI_TYPE_SINT64:
  case FFI_TYPE_DOUBLE:
    memcpy(&slot, value, 8);
    return slot;
  default:
    // int, int32, uint32, float, pointer: 4 bytes as they are
    memcpy(&slot, value, 4);
    return slot;
  }
}

/* Size of a variadic argument in the buffer, and its alignment. */
static void
vararg_layout(ffi_type *type, size_t *size, size_t *align)
{
  switch (type->type) {
  case FFI_TYPE_UINT8:
  case FFI_TYPE_SINT8:
    *size = *align = 1;
    return;
  case FFI_TYPE_UINT16:
  case FFI_TYPE_SINT16:
    *size = *align = 2;
    return;
  case FFI_TYPE_UINT64:
  case FFI_TYPE_SINT64:
  case FFI_TYPE_DOUBLE:
    *size = *align = 8;
    return;
  case FFI_TYPE_LONGDOUBLE:
    *size = 16;
    *align = 8;
    return;
  default:
    // int, uint32, sint32, float, pointer, and structs (as a pointer to a copy)
    *size = *align = 4;
    return;
  }
}

static uintptr_t
align_up(uintptr_t value, size_t align)
{
  return (value + align - 1) & ~(uintptr_t)(align - 1);
}

void
ffi_call(ffi_cif *cif, void (*fn)(void), void *rvalue, void **avalue)
{
  ffi_type *rtype = unbox(cif->rtype);
  // A small struct result is unboxed to a scalar, but the caller's buffer is
  // exactly the struct's size, so it must not get the widening to ffi_arg.
  int widen = cif->rtype->type != FFI_TYPE_STRUCT;
  int by_pointer = returns_by_pointer(rtype);
  int variadic = (cif->flags & VARARGS_FLAG) != 0;

  // At most: hidden result pointer, two slots per argument (long double), varargs pointer.
  size_t capacity = 2 + 2 * (size_t)cif->nargs;
  char signature[capacity + 1];
  uint64_t slots[capacity];
  size_t count = 0;
  size_t length = 0;

  if (by_pointer) {
    signature[length++] = 'v';
    if (rvalue == NULL)
      rvalue = __builtin_alloca(rtype->size);
    signature[length++] = 'i';
    slots[count++] = (uint32_t)(uintptr_t)rvalue;
  } else if (rtype->type == FFI_TYPE_VOID) {
    signature[length++] = 'v';
  } else {
    signature[length++] = letter_of(rtype->type);
  }

  for (unsigned i = 0; i < cif->nfixedargs; i++) {
    ffi_type *type = unbox(cif->arg_types[i]);
    switch (type->type) {
    case FFI_TYPE_VOID:
      break;
    case FFI_TYPE_LONGDOUBLE:
      signature[length++] = 'j';
      signature[length++] = 'j';
      memcpy(&slots[count], avalue[i], 16);
      count += 2;
      break;
    case FFI_TYPE_STRUCT: {
      // By value: the callee gets a pointer to a copy it may modify.
      void *copy = __builtin_alloca(type->size + type->alignment);
      copy = (void *)align_up((uintptr_t)copy, type->alignment);
      memcpy(copy, avalue[i], type->size);
      signature[length++] = 'i';
      slots[count++] = (uint32_t)(uintptr_t)copy;
      break;
    }
    default:
      signature[length++] = letter_of(type->type);
      slots[count++] = load_scalar(type->type, avalue[i]);
      break;
    }
  }

  if (variadic) {
    size_t total = 0;
    for (unsigned i = cif->nfixedargs; i < cif->nargs; i++) {
      size_t size, align;
      vararg_layout(unbox(cif->arg_types[i]), &size, &align);
      total = align_up(total, align) + size;
    }
    unsigned char *buffer = __builtin_alloca(total + 16);
    buffer = (unsigned char *)align_up((uintptr_t)buffer, 16);
    size_t offset = 0;
    for (unsigned i = cif->nfixedargs; i < cif->nargs; i++) {
      ffi_type *type = unbox(cif->arg_types[i]);
      size_t size, align;
      vararg_layout(type, &size, &align);
      offset = align_up(offset, align);
      if (type->type == FFI_TYPE_STRUCT) {
        void *copy = __builtin_alloca(type->size + type->alignment);
        copy = (void *)align_up((uintptr_t)copy, type->alignment);
        memcpy(copy, avalue[i], type->size);
        uint32_t pointer = (uint32_t)(uintptr_t)copy;
        memcpy(buffer + offset, &pointer, 4);
      } else {
        memcpy(buffer + offset, avalue[i], size);
      }
      offset += size;
    }
    signature[length++] = 'i';
    slots[count++] = (uint32_t)(uintptr_t)buffer;
  }

  uint64_t result = 0;
  __browseros_ffi_call((uint32_t)(uintptr_t)fn, signature, (uint32_t)length, slots, &result);

  if (by_pointer || rvalue == NULL)
    return;

  // Integral results narrower than a register are widened to ffi_arg, as every native port does.
  switch (rtype->type) {
  case FFI_TYPE_VOID:
    break;
  case FFI_TYPE_UINT8:
    if (widen)
      *(ffi_arg *)rvalue = (uint8_t)result;
    else
      *(uint8_t *)rvalue = (uint8_t)result;
    break;
  case FFI_TYPE_SINT8:
    if (widen)
      *(ffi_sarg *)rvalue = (int8_t)result;
    else
      *(int8_t *)rvalue = (int8_t)result;
    break;
  case FFI_TYPE_UINT16:
    if (widen)
      *(ffi_arg *)rvalue = (uint16_t)result;
    else
      *(uint16_t *)rvalue = (uint16_t)result;
    break;
  case FFI_TYPE_SINT16:
    if (widen)
      *(ffi_sarg *)rvalue = (int16_t)result;
    else
      *(int16_t *)rvalue = (int16_t)result;
    break;
  case FFI_TYPE_UINT64:
  case FFI_TYPE_SINT64:
  case FFI_TYPE_DOUBLE:
    memcpy(rvalue, &result, 8);
    break;
  default:
    // int, int32, uint32, float, pointer
    memcpy(rvalue, &result, 4);
    break;
  }
}

/*
 * Closures. ffi_closure_alloc reserves a slot in the indirect function table
 * (so the function pointer is known before ffi_prep_closure_loc, as libffi
 * promises) and ffi_prep_closure_loc asks the host to put a function of the
 * cif's lowered type there.
 */

void *__attribute__((visibility("default")))
ffi_closure_alloc(size_t size, void **code)
{
  ffi_closure *closure = malloc(size);
  if (closure == NULL)
    return NULL;
  int32_t index = __browseros_ffi_closure_reserve();
  if (index <= 0) {
    free(closure);
    return NULL;
  }
  // The trampoline field holds the table index (FFI_TRAMPOLINE_SIZE is 4).
  memcpy(closure->tramp, &index, sizeof index);
  *code = (void *)(uintptr_t)index;
  return closure;
}

void __attribute__((visibility("default")))
ffi_closure_free(void *pointer)
{
  ffi_closure *closure = pointer;
  if (closure == NULL)
    return;
  int32_t index;
  memcpy(&index, closure->tramp, sizeof index);
  __browseros_ffi_closure_release((uint32_t)index);
  free(closure);
}

ffi_status
ffi_prep_closure_loc(ffi_closure *closure, ffi_cif *cif,
                     void (*fun)(ffi_cif *, void *, void **, void *),
                     void *user_data, void *codeloc)
{
  if (cif->abi != FFI_WASM32)
    return FFI_BAD_ABI;
  ffi_type *rtype = unbox(cif->rtype);
  size_t capacity = 2 + 2 * (size_t)cif->nargs;
  char signature[capacity + 1];
  size_t length = 0;

  if (returns_by_pointer(rtype)) {
    signature[length++] = 'v';
    signature[length++] = 'i';
  } else if (rtype->type == FFI_TYPE_VOID) {
    signature[length++] = 'v';
  } else {
    signature[length++] = letter_of(rtype->type);
  }
  for (unsigned i = 0; i < cif->nfixedargs; i++) {
    ffi_type *type = unbox(cif->arg_types[i]);
    switch (type->type) {
    case FFI_TYPE_VOID:
      break;
    case FFI_TYPE_LONGDOUBLE:
      signature[length++] = 'j';
      signature[length++] = 'j';
      break;
    case FFI_TYPE_STRUCT:
      signature[length++] = 'i';
      break;
    default:
      signature[length++] = letter_of(type->type);
      break;
    }
  }
  if (cif->flags & VARARGS_FLAG)
    signature[length++] = 'i';

  closure->cif = cif;
  closure->fun = fun;
  closure->user_data = user_data;
  if (__browseros_ffi_closure_set((uint32_t)(uintptr_t)codeloc, signature, (uint32_t)length, closure) != 0)
    return FFI_BAD_TYPEDEF;
  return FFI_OK;
}

/*
 * Called by the host's function for a closure. `slots` holds the lowered
 * arguments in the order of the closure's signature (8 bytes each) and
 * `result` is an 8-byte buffer the host reads back for a scalar result.
 */
__attribute__((export_name("__browseros_ffi_closure_call"))) void
__browseros_ffi_closure_call(ffi_closure *closure, uint64_t *slots, void *result)
{
  ffi_cif *cif = closure->cif;
  ffi_type *rtype = unbox(cif->rtype);
  size_t next = 0;
  void *rvalue = result;
  if (returns_by_pointer(rtype))
    rvalue = (void *)(uintptr_t)(uint32_t)slots[next++];

  void *avalue[cif->nargs > 0 ? cif->nargs : 1];
  for (unsigned i = 0; i < cif->nfixedargs; i++) {
    ffi_type *type = unbox(cif->arg_types[i]);
    switch (type->type) {
    case FFI_TYPE_VOID:
      avalue[i] = NULL;
      break;
    case FFI_TYPE_LONGDOUBLE:
      // Two consecutive slots: the low and high halves, in place.
      avalue[i] = &slots[next];
      next += 2;
      break;
    case FFI_TYPE_STRUCT: {
      // The caller passed a pointer to its copy; the function gets one of its own.
      void *source = (void *)(uintptr_t)(uint32_t)slots[next++];
      void *copy = __builtin_alloca(type->size + type->alignment);
      copy = (void *)align_up((uintptr_t)copy, type->alignment);
      memcpy(copy, source, type->size);
      avalue[i] = copy;
      break;
    }
    default:
      // Little-endian: the value is at the start of its slot.
      avalue[i] = &slots[next++];
      break;
    }
  }
  if (cif->flags & VARARGS_FLAG) {
    unsigned char *buffer = (unsigned char *)(uintptr_t)(uint32_t)slots[next++];
    size_t offset = 0;
    for (unsigned i = cif->nfixedargs; i < cif->nargs; i++) {
      ffi_type *type = unbox(cif->arg_types[i]);
      size_t size, align;
      vararg_layout(type, &size, &align);
      offset = align_up(offset, align);
      if (type->type == FFI_TYPE_STRUCT) {
        uint32_t pointer;
        memcpy(&pointer, buffer + offset, 4);
        void *copy = __builtin_alloca(type->size + type->alignment);
        copy = (void *)align_up((uintptr_t)copy, type->alignment);
        memcpy(copy, (void *)(uintptr_t)pointer, type->size);
        avalue[i] = copy;
      } else {
        avalue[i] = buffer + offset;
      }
      offset += size;
    }
  }

  closure->fun(cif, rvalue, avalue, closure->user_data);
}
