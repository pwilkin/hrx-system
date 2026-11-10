// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdlib>
#include <cstring>

#include "iree/net/carrier/tcp/factory.h"
#include "iree/net/cts/transport_backend.h"

namespace iree::net::cts {
namespace {

iree_status_t CreateFactory(iree_allocator_t host_allocator,
                            iree_net_transport_factory_t** out_factory) {
  auto options = iree_net_tcp_factory_options_default();
  const char* disable_zero_copy =
      std::getenv("IREE_NET_CTS_TCP_DISABLE_ZERO_COPY");
  if (disable_zero_copy && std::strcmp(disable_zero_copy, "1") == 0) {
    options.connection_options.carrier_options.zero_copy_min_send_size =
        IREE_HOST_SIZE_MAX;
  }
#if defined(IREE_NET_CTS_TCP_RECEIVE_PLACEMENT)
  options.connection_options.receive_mode = IREE_NET_TCP_RECEIVE_MODE_PLACEMENT;
#endif
  return iree_net_tcp_factory_create(&options, host_allocator, out_factory);
}

iree_status_t MakeBindAddress(std::string* out_address) {
  *out_address = "127.0.0.1:0";
  return iree_ok_status();
}

}  // namespace

const TransportBackend& GetTransportBackend() {
  static const TransportBackend backend = {
      .name = "tcp",
      .required_capabilities = IREE_NET_TRANSPORT_CAPABILITY_RELIABLE |
                               IREE_NET_TRANSPORT_CAPABILITY_ORDERED,
      .create_factory = CreateFactory,
      .make_bind_address = MakeBindAddress,
  };
  return backend;
}

}  // namespace iree::net::cts
