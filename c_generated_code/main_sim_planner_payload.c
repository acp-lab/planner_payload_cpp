/*
 * Copyright (c) The acados authors.
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */


// standard
#include <stdio.h>
#include <stdlib.h>
// acados
#include "acados/utils/print.h"
#include "acados/utils/math.h"
#include "acados_c/sim_interface.h"
#include "acados_sim_solver_planner_payload.h"

#define NX     PLANNER_PAYLOAD_NX
#define NZ     PLANNER_PAYLOAD_NZ
#define NU     PLANNER_PAYLOAD_NU
#define NP     PLANNER_PAYLOAD_NP


int main()
{
    int status = 0;
    planner_payload_sim_solver_capsule *capsule = planner_payload_acados_sim_solver_create_capsule();
    status = planner_payload_acados_sim_create(capsule);

    if (status)
    {
        printf("acados_create() returned status %d. Exiting.\n", status);
        exit(1);
    }

    sim_config *acados_sim_config = planner_payload_acados_get_sim_config(capsule);
    sim_in *acados_sim_in = planner_payload_acados_get_sim_in(capsule);
    sim_out *acados_sim_out = planner_payload_acados_get_sim_out(capsule);
    void *acados_sim_dims = planner_payload_acados_get_sim_dims(capsule);

    // initial condition
    double x_current[NX];
    x_current[0] = 0.0;
    x_current[1] = 0.0;
    x_current[2] = 0.0;
    x_current[3] = 0.0;
    x_current[4] = 0.0;
    x_current[5] = 0.0;
    x_current[6] = 0.0;
    x_current[7] = 0.0;
    x_current[8] = 0.0;
    x_current[9] = 0.0;
    x_current[10] = 0.0;
    x_current[11] = 0.0;

  
    x_current[0] = 0;
    x_current[1] = 0;
    x_current[2] = 0.47;
    x_current[3] = 0;
    x_current[4] = 0;
    x_current[5] = 0;
    x_current[6] = -0;
    x_current[7] = -0;
    x_current[8] = -1;
    x_current[9] = 0;
    x_current[10] = 0;
    x_current[11] = 0;
    
  


    // initial value for control input
    double u0[NU];
    u0[0] = 0.0;
    u0[1] = 0.0;
    u0[2] = 0.0;
    u0[3] = 0.0;
    // set parameters
    double p[NP];
    p[0] = 0;
    p[1] = 0;
    p[2] = 0.47;
    p[3] = 0;
    p[4] = 0;
    p[5] = 0;
    p[6] = -0;
    p[7] = -0;
    p[8] = -1;
    p[9] = 0;
    p[10] = 0;
    p[11] = 0;
    p[12] = 1.106568;
    p[13] = 0;
    p[14] = 0;
    p[15] = 0;
    p[16] = 210;
    p[17] = 210;
    p[18] = 210;
    p[19] = 1;
    p[20] = 1;
    p[21] = 1;
    p[22] = 5;
    p[23] = 5;
    p[24] = 5;
    p[25] = 1;
    p[26] = 1;
    p[27] = 1;
    p[28] = 210;
    p[29] = 210;
    p[30] = 210;
    p[31] = 1;
    p[32] = 1;
    p[33] = 1;
    p[34] = 5;
    p[35] = 5;
    p[36] = 5;
    p[37] = 1;
    p[38] = 1;
    p[39] = 1;
    p[40] = 0.5;
    p[41] = 0.1;
    p[42] = 0.1;
    p[43] = 0.1;

    planner_payload_acados_sim_update_params(capsule, p, NP);
  

  


    int n_sim_steps = 3;
    // solve ocp in loop
    for (int ii = 0; ii < n_sim_steps; ii++)
    {
        // set inputs
        sim_in_set(acados_sim_config, acados_sim_dims,
            acados_sim_in, "x", x_current);
        sim_in_set(acados_sim_config, acados_sim_dims,
            acados_sim_in, "u", u0);

        // solve
        status = planner_payload_acados_sim_solve(capsule);
        if (status != ACADOS_SUCCESS)
        {
            printf("acados_solve() failed with status %d.\n", status);
        }

        // get outputs
        sim_out_get(acados_sim_config, acados_sim_dims,
               acados_sim_out, "x", x_current);

    

        // print solution
        printf("\nx_current, %d\n", ii);
        for (int jj = 0; jj < NX; jj++)
        {
            printf("%e\n", x_current[jj]);
        }
    }

    printf("\nPerformed %d simulation steps with acados integrator successfully.\n\n", n_sim_steps);

    // free solver
    status = planner_payload_acados_sim_free(capsule);
    if (status) {
        printf("planner_payload_acados_sim_free() returned status %d. \n", status);
    }

    planner_payload_acados_sim_solver_free_capsule(capsule);

    return status;
}
